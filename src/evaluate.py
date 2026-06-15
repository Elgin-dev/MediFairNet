# /home/elgin-dev/medifairnet/src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def run_evaluation_suite(model_encoder, model_classifier, dataloader):
    """
    Evaluates the trained model for both diagnostic accuracy (AUROC) 
    and demographic equity across patient cohorts.
    """
    print("\n📊 Initializing Clinical Validation & Bias Audit Engine...")
    
    # Force evaluation mode (disables standard dropout, though MC dropout remains manual)
    model_encoder.eval()
    model_classifier.eval()
    
    all_targets = []
    all_preds = []
    all_demographics = []
    all_uncertainties = []
    
    with torch.no_grad():
        for images, disease_targets, demographic_targets, text_descriptions in dataloader:
            # 1. Forward Pass
            z_spec, _ = model_encoder(images)
            
            # 2. Extract calibrated predictions and uncertainty via MC Dropout simulation
            mean_probs, uncertainty_variance = model_classifier.predict_with_uncertainty(z_spec, num_samples=10)
            
            # Collect matrices for global evaluation
            all_targets.append(disease_targets.cpu().numpy())
            all_preds.append(mean_probs.cpu().numpy())
            all_demographics.append(demographic_targets.cpu().numpy())
            all_uncertainties.append(uncertainty_variance.cpu().numpy())
            
    # Concatenate all collected batch matrices
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    all_demographics = np.concatenate(all_demographics)
    all_uncertainties = np.vstack(all_uncertainties)
    
    # 3. Compute Mean Macro AUROC (Diagnostic Accuracy Check)
    try:
        macro_auroc = roc_auc_score(all_targets, all_preds, average='macro')
    except ValueError:
        # Fallback if mock dataset lacks positive instances for some classes
        macro_auroc = 0.5000
        
    # 4. Compute Demographic Equity Disparity (Bias Verification Check)
    # We measure performance variance across the 4 demographic cohorts
    cohort_scores = []
    for cohort_idx in range(4):
        cohort_mask = (all_demographics == cohort_idx)
        if np.sum(cohort_mask) > 0:
            try:
                cohort_auroc = roc_auc_score(all_targets[cohort_mask], all_preds[cohort_mask], average='macro')
                cohort_scores.append(cohort_auroc)
            except ValueError:
                cohort_scores.append(0.5000)
        else:
            cohort_scores.append(0.5000)
            
    # Max performance disparity between best and worst treated demographic groups
    fairness_disparity = np.max(cohort_scores) - np.min(cohort_scores)
    
    print("\n================ MEDIFAIRNET VALIDATION REPORT ================")
    print(f"🏥 Mean Clinical Diagnostic AUROC:    {macro_auroc:.4f}  (Target: >0.85)")
    print(f"⚖️ Demographic Performance Disparity: {fairness_disparity:.4f}  (Target: <0.05)")
    print(f"🛡️ Average Epistemic Uncertainty:    {np.mean(all_uncertainties):.6f}")
    print("===============================================================")
    
    return macro_auroc, fairness_disparity

if __name__ == "__main__":
    # Test execution harness using standard mock streaming structures
    from data_loaders.dataset import MediFairXRayDataset
    from torch.utils.data import DataLoader
    # FIX: Import the missing encoder class cleanly
    from models.encoders import FeatureDisentanglementEncoder
    from models.uncertainty import TrustCalibratedClassifier
    
    DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule']
    
    dataset = MediFairXRayDataset(
        csv_file="data/nih_manifest.csv",
        img_dir="data/images/",
        disease_cols=DISEASES,
        demographic_col='demographic_group_idx'
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Temporary test instances
    encoder = FeatureDisentanglementEncoder(class_spec_dim=256, class_agn_dim=128)
    classifier = TrustCalibratedClassifier(feature_dim=256, num_classes=len(DISEASES))
    
    run_evaluation_suite(encoder, classifier, dataloader)