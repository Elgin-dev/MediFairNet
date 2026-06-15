# /home/elgin-dev/medifairnet/src/run_training.py

import torch
import os
import sys
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  
from data_loaders.dataset import MediFairXRayDataset
from models.encoders import FeatureDisentanglementEncoder, AdversarialDomainClassifier
from models.fusion import BioBertCrossAttentionFusion
from models.uncertainty import TrustCalibratedClassifier
from losses.disentangle_loss import MediFairNetLossEngine

def execute_actual_training():
    print("静态 Verification Checkpoints Initializing...")
    
    csv_path = os.path.abspath("data/nih_manifest.csv")
    img_dir_path = os.path.abspath("data/images/")
    
    if not os.path.exists(csv_path) or not os.path.exists(img_dir_path):
        print("❌ ERROR: Missing local file system data assets!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Execution hardware target selected: {device}")

    writer = SummaryWriter(log_dir="data/tb_logs")

    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 1e-4
    DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule']
    
    dataset = MediFairXRayDataset(csv_file=csv_path, img_dir=img_dir_path, disease_cols=DISEASES, demographic_col='demographic_group_idx')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"📦 Successfully parsed {len(dataset)} samples into {len(dataloader)} batches.")

    encoder = FeatureDisentanglementEncoder().to(device)
    domain_classifier = AdversarialDomainClassifier(num_demographics=2).to(device)
    fusion_engine = BioBertCrossAttentionFusion().to(device)
    classifier = TrustCalibratedClassifier(num_classes=len(DISEASES)).to(device)
    loss_engine = MediFairNetLossEngine().to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(classifier.parameters()) + list(domain_classifier.parameters()),
        lr=LEARNING_RATE
    )

    global_step = 0
    print("\n🚀 Training pipeline launched silently. Check TensorBoard for real-time live graphs!")
    
    for epoch in range(EPOCHS):
        encoder.train()
        classifier.train()
        
        epoch_start_time = time.time()
        running_total_loss = 0.0
        running_task_loss = 0.0
        running_bias_penalty = 0.0
        
        for batch_idx, (images, disease_targets, demographic_targets, text_descriptions) in enumerate(dataloader):
            images, disease_targets, demographic_targets = images.to(device), disease_targets.to(device), demographic_targets.to(device)
            
            optimizer.zero_grad()
            z_spec, z_agn = encoder(images)
            demog_logits = domain_classifier(z_spec)
            fused_features = fusion_engine(z_spec, text_descriptions)
            raw_logits = classifier.forward_logits(fused_features)
            
            total_loss, l_task, l_ortho, l_bias = loss_engine(
                logits=raw_logits, targets=disease_targets,
                z_spec=z_spec, z_agn=z_agn,
                demographic_logits=demog_logits, demographic_labels=demographic_targets
            )
            
            total_loss.backward()
            optimizer.step()
            
            # 📊 Quietly send everything to TensorBoard in the background
            writer.add_scalar("Batch/Total_Loss", total_loss.item(), global_step)
            writer.add_scalar("Batch/Clinical_Task_Loss", l_task.item(), global_step)
            writer.add_scalar("Batch/Demographic_Bias_Penalty", l_bias.item(), global_step)
            writer.flush()
            
            # Accumulate metrics for the terminal summary box
            running_total_loss += total_loss.item()
            running_task_loss += l_task.item()
            running_bias_penalty += l_bias.item()
            global_step += 1

        # 📈 Calculate Epoch Averages
        avg_total = running_total_loss / len(dataloader)
        avg_task = running_task_loss / len(dataloader)
        avg_bias = running_bias_penalty / len(dataloader)
        elapsed = time.time() - epoch_start_time

        # 📊 BEAUTIFUL SUMMARY BOX PER EPOCH
        print("="*60)
        print(f"🏁 EPOCH {epoch + 1}/{EPOCHS} COMPLETE ({elapsed:.1f}s)")
        print(f" ↳ 🌌 Avg Total System Loss: {avg_total:.4f}")
        print(f" ↳ 🏥 Avg Clinical Task Loss: {avg_task:.4f}")
        print(f" ↳ ⚖️ Avg Fairness Bias Penalty: {avg_bias:.4f}")
        print("="*60)

        # 🧬 INTERPRETABILITY LOGGING
        for name, param in encoder.named_parameters():
            if "weight" in name:
                writer.add_histogram(f"EncoderWeights/{name}", param, epoch)

    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'classifier_state_dict': classifier.state_dict()
    }, "medifairnet_core.pt")
    
    writer.close()  
    print("\n✅ All training epochs finished! Weights saved to 'medifairnet_core.pt'.")

if __name__ == "__main__":
    execute_actual_training()