# /home/elgin-dev/medifairnet/src/losses/disentangle_loss.py

import torch
import torch.nn as nn

class MediFairNetLossEngine(nn.Module):
    """
    Optimizes multi-label classification accuracy while enforcing 
    feature orthogonality and minimizing demographic data leakage.
    """
    def __init__(self, z_spec_dim: int = 256, z_agn_dim: int = 128):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Linear projection to align dimensions for orthogonality calculation
        self.projection_align = nn.Linear(z_agn_dim, z_spec_dim)

    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor, 
                z_spec: torch.Tensor, 
                z_agn: torch.Tensor, 
                demographic_logits: torch.Tensor, 
                demographic_labels: torch.Tensor,
                alpha: float = 0.5,
                beta: float = 0.2) -> tuple:
        """
        Args:
            logits: Classifier outputs [Batch, Num_Classes]
            targets: True disease binary vectors [Batch, Num_Classes]
            z_spec: Pathology embedding vector [Batch, 256]
            z_agn: Anatomy structural embedding vector [Batch, 128]
            demographic_logits: Bias classifier predictions [Batch, Num_Demographics]
            demographic_labels: Target demographic classes [Batch]
        """
        # 1. Primary Disease Diagnosis Loss
        l_task = self.bce_loss(logits, targets)
        
        # 2. Project z_agn to match z_spec's dimension for element-wise compatibility
        z_agn_projected = self.projection_align(z_agn)
        
        # Structural Orthogonality Loss (Forces Cosine Similarity toward 0)
        z_spec_norm = nn.functional.normalize(z_spec, p=2, dim=1)
        z_agn_norm = nn.functional.normalize(z_agn_projected, p=2, dim=1)
        
        # Now shapes match completely: [Batch, 256] * [Batch, 256]
        dot_product = torch.sum(z_spec_norm * z_agn_norm, dim=1)
        l_ortho = torch.mean(torch.pow(dot_product, 2))
        
        # 3. Demographic Adversarial Loss
        l_bias = self.ce_loss(demographic_logits, demographic_labels)
        
        # Total Consolidated Loss Equation
        total_loss = l_task + (alpha * l_ortho) - (beta * l_bias)
        
        return total_loss, l_task, l_ortho, l_bias