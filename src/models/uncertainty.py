# /home/elgin-dev/medifairnet/src/models/uncertainty.py

import torch
import torch.nn as nn

class TrustCalibratedClassifier(nn.Module):
    """
    Multi-label clinical classifier equipped with Monte Carlo Dropout 
    sampling for uncertainty extraction and temperature scaling hooks.
    """
    def __init__(self, feature_dim: int = 256, num_classes: int = 14):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Retained during inference for MC sampling
            nn.Linear(128, num_classes)
        )
        
        # Learnable Temperature parameter initialized to 1.0 (neutral scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw unscaled logits for calculation optimization."""
        return self.classifier(x)

    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 10):
        """
        Forces dropout status to active to sample multiple forward passes,
        deriving empirical confidence mean and epistemic variance.
        """
        # Explicitly force training mode active to keep dropout functional
        self.train()
        
        sampled_predictions = []
        
        with torch.set_grad_enabled(False):
            for _ in range(num_samples):
                logits = self.classifier(x)
                # Apply Temperature Scaling calculation to smooth calibration
                scaled_logits = logits / self.temperature
                # Map to multi-label probability configuration space
                probs = torch.sigmoid(scaled_logits)
                sampled_predictions.append(probs.unsqueeze(0))
        
        # Stack samples along a new prefix dimension: [Samples, Batch, Classes]
        sampled_predictions = torch.cat(sampled_predictions, dim=0)
        
        # Calculate statistical mean and variance across the samples
        mean_probabilities = sampled_predictions.mean(dim=0)
        uncertainty_variance = sampled_predictions.var(dim=0)
        
        return mean_probabilities, uncertainty_variance