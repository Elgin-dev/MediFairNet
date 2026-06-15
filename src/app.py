# /home/elgin-dev/medifairnet/src/app.py

import torch
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from PIL import Image
from torchvision.transforms import v2
from models.encoders import FeatureDisentanglementEncoder
from models.uncertainty import TrustCalibratedClassifier

app = FastAPI(
    title="MediFairNet API Core",
    description="Production endpoint for Fair Medical Imaging Diagnostics with Uncertainty Quantification.",
    version="1.0.0"
)

# 1. Initialize and lock the models on CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = FeatureDisentanglementEncoder(class_spec_dim=256, class_agn_dim=128).to(device)
classifier = TrustCalibratedClassifier(feature_dim=256, num_classes=6).to(device)

encoder.eval()
classifier.eval()

# 2. Standardized preprocessing transform pipeline
img_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule']

def verify_input_domain(pil_image: Image.Image) -> bool:
    """
    🛡️ GATEKEEPER LAYER (Out-of-Distribution Filter)
    Uses a Structural Variance Check to determine if the uploaded asset is a 
    valid medical radiograph or an invalid document/screenshot.
    """
    # Convert image to grayscale numpy array for distribution analysis
    gray_img = pil_image.convert("L")
    img_array = np.array(gray_img, dtype=np.float32) / 255.0
    
    # Calculate overall image standard deviation
    global_std = np.std(img_array)
    
    # Split image into a 3x3 grid to check structural symmetry (X-rays are centered)
    h, w = img_array.shape
    center_block = img_array[h//3:2*h//3, w//3:2*w//3]
    center_mean = np.mean(center_block)
    
    # 🔍 Out-Of-Distribution Threshold Logic:
    # Text documents/screenshots have massive contrast variance (sharp black text on pure white backgrounds, global_std > 0.38)
    # OR they are incredibly uniform web layouts (global_std < 0.12)
    # True lung X-rays hover beautifully inside a mid-tone structural variance valley (typically 0.15 to 0.35)
    if global_std > 0.38 or global_std < 0.12:
        return False
        
    # Text pages/spreadsheets have incredibly bright centers. True lung cavities are dark/mid-toned.
    if center_mean > 0.82:
        return False
        
    return True

@app.post("/v1/predict")
async def predict_xray(file: UploadFile = File(...)):
    """
    Accepts a chest X-ray image file, runs an Out-of-Distribution validation check, 
    processes it through the Tri-Fusion system, and returns debiased diagnostics.
    """
    # Read the incoming image stream safely
    image_bytes = await file.read()
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted file format. Please upload a clean image asset."
        )
    
    # 🛑 CRITICAL GATEKEEPER INSPECTION
    if not verify_input_domain(image):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="⚠️ OOD ALIGNMENT ERROR: The uploaded asset failed our structural radiograph validation check. "
                   "Please verify you are uploading a valid chest X-ray and not a document, text file, or UI screenshot."
        )
    
    # Preprocess and add a batch dimension: [1, 3, 224, 224]
    input_tensor = img_transform(image).unsqueeze(0).to(device)
    
    # 3. Process through the MediFairNet pipeline
    with torch.no_grad():
        z_spec, _ = encoder(input_tensor)
        # Use Monte Carlo sampling to extract predictions and variance
        mean_probs, uncertainty_variance = classifier.predict_with_uncertainty(z_spec, num_samples=10)
        
    # 4. Formulate the JSON response payload structure
    probabilities = mean_probs.squeeze(0).cpu().numpy()
    uncertainties = uncertainty_variance.squeeze(0).cpu().numpy()
    
    diagnostic_report = {}
    for idx, disease in enumerate(DISEASES):
        diagnostic_report[disease] = {
            "probability": float(probabilities[idx]),
            "epistemic_uncertainty": float(uncertainties[idx]),
            "status": "High Risk / Review Required" if probabilities[idx] > 0.5 else "Clear"
        }
        
    return {
        "status": "SUCCESS",
        "model_metadata": {
            "backbone": "DenseNet121",
            "debiasing_active": True,
            "uncertainty_method": "Monte Carlo Dropout (10 Samples)"
        },
        "diagnostics": diagnostic_report
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)