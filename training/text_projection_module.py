import torch
import torch.nn as nn

class ProjectionModule(nn.Module):
    def __init__(self, bert_dim=768, clip_dim=512):
        """
        Initializes a learnable projection layer that maps BERT embeddings 
        to CLIP embedding space.

        Args:
            bert_dim (int): Dimensionality of BERT embeddings (default: 768).
            clip_dim (int): Dimensionality of CLIP embeddings (default: 512).
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, 1024),  # Expands features
            nn.ReLU(),  # Non-linearity to capture relationships
            nn.Linear(1024, clip_dim)  # Projects down to CLIP space
        )

    def forward(self, bert_embedding):
        """
        Projects a BERT embedding into CLIP’s embedding space.

        Args:
            bert_embedding (torch.Tensor): Input BERT embedding.

        Returns:
            torch.Tensor: Projected embedding aligned with CLIP space.
        """
        return self.projection(bert_embedding)  # Apply linear projection
