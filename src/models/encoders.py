# medifairnet/src/models/encoders.py

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class FeatureDisentanglementEncoder(nn.Module):
    """
    Decomposes medical image features into clinical disease-specific patterns
    and structural class-agnostic anatomy. Inspired by IEEE TMI 2025.
    """
    def __init__(self, class_spec_dim: int = 256, class_agn_dim: int = 128):
        super().__init__()
        
        # Load a modern, stable DenseNet121 optimized for radiology data
        weights = DenseNet121_Weights.DEFAULT
        backbone = densenet121(weights=weights)
        
        # Extract features directly from the convolutional blocks
        self.shared_backbone = backbone.features
        num_features = backbone.classifier.in_features  # Dynamically fetches 1024
        # Branch A: Pathological / Disease Pattern Feature Space
        self.class_specific_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, class_spec_dim)
        )
        
        # Branch B: Structural / Anatomical Feature Space
        self.class_agnostic_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, class_agn_dim)
        )

    def forward(self, x: torch.Tensor):
        # Extract 2D feature maps from backbone
        features = self.shared_backbone(x)
        
        # Global Average Pooling to collapse spatial dimensions to [Batch, 1024]
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Projection into the isolated latent spaces
        z_spec = self.class_specific_head(features)
        z_agn = self.class_agnostic_head(features)
        
        return z_spec, z_agn


class AdversarialDomainClassifier(nn.Module):
    """
    Used to intercept and eliminate demographic data signatures from the 
    class-specific feature representations using adversarial domain training.
    """
    def __init__(self, input_dim: int = 256, num_demographics: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_demographics)
        )
        
    def forward(self, z_spec: torch.Tensor) -> torch.Tensor:
        # Predicts demographic protected variables (e.g., race, biological sex)
        return self.classifier(z_spec)