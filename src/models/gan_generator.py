# /home/elgin-dev/medifairnet/src/models/gan_generator.py

import torch
import torch.nn as nn

class WGANFeatureGenerator(nn.Module):
    """
    Conditional Generator that synthesizes realistic visual feature vectors 
    conditioned on BioBERT clinical semantic embeddings.
    """
    def __init__(self, noise_dim: int = 128, text_dim: int = 256, visual_feature_dim: int = 256):
        super().__init__()
        
        # Input dim is noise vector + projected text embedding vector
        input_dim = noise_dim + text_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # WGAN-GP prefers LayerNorm over BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, visual_feature_dim),
            # No activation at the output layer because we are synthesizing 
            # un-bounded hidden representations (z_spec) directly
        )

    def forward(self, noise: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Concatenate latent noise with the conditional text embedding
        x = torch.cat([noise, text_emb], dim=-1)
        return self.net(x)


class WGANFeatureCritic(nn.Module):
    """
    WGAN Critic (Discriminator) that assesses the quality/authenticity of 
    a visual feature representation relative to its matching text context.
    """
    def __init__(self, visual_feature_dim: int = 256, text_dim: int = 256):
        super().__init__()
        
        input_dim = visual_feature_dim + text_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1) # Outputs a scalar score (not a probability) for WGAN
        )

    def forward(self, visual_features: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # Concatenate the feature vector with its conditional text attribute
        x = torch.cat([visual_features, text_emb], dim=-1)
        return self.net(x)