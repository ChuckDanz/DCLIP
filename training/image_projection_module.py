# image_projection_module.py
import torch
import torch.nn as nn

class ImageProjectionModule(nn.Module):
    def __init__(self, clip_dim=512, hidden_dim=1024):
        """
        Initializes a projection layer for image embeddings.
        
        Args:
            clip_dim (int): Dimensionality of CLIP image embeddings.
            hidden_dim (int): Intermediate hidden layer size.
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(clip_dim + 4, hidden_dim),  # +4 for bbox coordinates
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, clip_dim)
        )

    def forward(self, context_features, positions):
        """
        Projects a CLIP image embedding into a refined embedding space.
        
        Args:
            context_features (torch.Tensor): Input CLIP image embedding.
            positions (torch.Tensor): Bounding box coordinates.
            
        Returns:
            torch.Tensor: Projected image embedding.
        """
        # Concatenate context features with position
        x = torch.cat([context_features, positions], dim=1)
        return self.projection(x)
