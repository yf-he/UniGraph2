from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from transformers import AutoModel, ViTModel

from .encoders import GraphEncoder
from .moe import MixtureOfExperts


class UniGraph2Model(nn.Module):
    """UniGraph2: A unified multimodal graph foundation model."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize modality-specific encoders
        self.text_encoder = self._init_text_encoder()
        self.image_encoder = self._init_image_encoder()
        self.graph_encoder = self._init_graph_encoder()
        
        # Projection layers to common dimension
        hidden_dim = config["model"]["hidden_dim"]
        self.text_proj = nn.Linear(768, hidden_dim)  # BERT hidden size
        self.image_proj = nn.Linear(768, hidden_dim)  # ViT hidden size
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Mixture of Experts for feature alignment
        self.moe = MixtureOfExperts(
            input_dim=hidden_dim,
            num_experts=config["model"]["moe"]["num_experts"],
            top_k=config["model"]["moe"]["top_k"],
            capacity_factor=config["model"]["moe"]["capacity_factor"]
        )
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _init_text_encoder(self) -> nn.Module:
        """Initialize the text encoder."""
        model_name = self.config["model"]["text_encoder"]["model_name"]
        trainable = self.config["model"]["text_encoder"]["trainable"]
        
        encoder = AutoModel.from_pretrained(model_name)
        if not trainable:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder
    
    def _init_image_encoder(self) -> nn.Module:
        """Initialize the image encoder."""
        model_name = self.config["model"]["image_encoder"]["model_name"]
        trainable = self.config["model"]["image_encoder"]["trainable"]
        
        encoder = ViTModel.from_pretrained(model_name)
        if not trainable:
            for param in encoder.parameters():
                param.requires_grad = False
        return encoder
    
    def _init_graph_encoder(self) -> nn.Module:
        """Initialize the graph encoder."""
        return GraphEncoder(self.config["model"]["graph_encoder"])
    
    def encode_text(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode text inputs."""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use CLS token embedding
        text_emb = outputs.last_hidden_state[:, 0]
        return self.text_proj(text_emb)
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode image inputs."""
        outputs = self.image_encoder(pixel_values=pixel_values)
        # Use CLS token embedding
        image_emb = outputs.last_hidden_state[:, 0]
        return self.image_proj(image_emb)
    
    def encode_graph(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Encode graph inputs."""
        graph_emb = self.graph_encoder(graph)
        return self.graph_proj(graph_emb)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        graph: Optional[Union[Data, Batch]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        embeddings = {}
        
        # Encode available modalities
        if input_ids is not None and attention_mask is not None:
            embeddings["text"] = self.encode_text(input_ids, attention_mask)
            
        if pixel_values is not None:
            embeddings["image"] = self.encode_image(pixel_values)
            
        if graph is not None:
            embeddings["graph"] = self.encode_graph(graph)
        
        # Apply MoE for feature alignment
        if len(embeddings) > 0:
            # Stack all embeddings
            stacked_emb = torch.stack(list(embeddings.values()), dim=1)
            # Apply MoE
            aligned_emb = self.moe(stacked_emb)
            # Apply layer norm
            unified_emb = self.layer_norm(aligned_emb)
            
            # Update embeddings with aligned versions
            for i, key in enumerate(embeddings.keys()):
                embeddings[key] = unified_emb[:, i]
        
        return embeddings
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "UniGraph2Model":
        """Load a pretrained model from a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint["config"]
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def save_pretrained(self, save_path: str):
        """Save the model to a checkpoint."""
        checkpoint = {
            "config": self.config,
            "model_state_dict": self.state_dict()
        }
        torch.save(checkpoint, save_path) 