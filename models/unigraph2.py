import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import dgl
import numpy as np


class MoE(nn.Module):
    """Mixture of Experts module for cross-domain and cross-modality alignment"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        num_selected_experts: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get expert weights
        weights = self.gate(x)
        
        # Select top-k experts
        top_weights, top_indices = torch.topk(weights, self.num_selected_experts, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Combine expert outputs
        selected_outputs = torch.gather(
            expert_outputs,
            1,
            top_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        )
        output = torch.sum(selected_outputs * top_weights.unsqueeze(-1), dim=1)
        
        return output


class DomainSpecificDecoder(nn.Module):
    """Decoder for specific graph domains"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class SPDDecoder(nn.Module):
    """Shortest Path Distance decoder"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_i, x_j], dim=-1)
        return self.decoder(x)


class UniGraph2(nn.Module):
    """UniGraph2 model for multimodal graph representation learning"""
    
    def __init__(
        self,
        input_dims: Dict[str, int],  # Dictionary of input dimensions for each modality
        hidden_dim: int = 768,
        num_experts: int = 8,
        num_selected_experts: int = 2,
        num_layers: int = 3,
        feat_drop_rate: float = 0.1,
        edge_mask_rate: float = 0.1,
        gamma: float = 2.0,
        lambda_spd: float = 0.5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_drop_rate = feat_drop_rate
        self.edge_mask_rate = edge_mask_rate
        self.gamma = gamma
        self.lambda_spd = lambda_spd
        
        # Mixture of Experts
        self.moe = MoE(hidden_dim, hidden_dim, num_experts, num_selected_experts)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            dgl.nn.GATConv(hidden_dim, hidden_dim, num_heads=4)
            for _ in range(num_layers)
        ])
        
        # Domain-specific decoders
        self.domain_decoders = nn.ModuleDict({
            domain: DomainSpecificDecoder(hidden_dim, input_dims[domain])
            for domain in input_dims.keys()
        })
        
        # SPD decoder
        self.spd_decoder = SPDDecoder(hidden_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        
    def _mask_features(
        self,
        features: torch.Tensor,
        mask_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask node features"""
        num_nodes = features.size(0)
        num_masked = int(num_nodes * mask_rate)
        
        # Create mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[:num_masked] = True
        mask = mask[torch.randperm(num_nodes)]
        
        # Apply mask
        masked_features = features.clone()
        masked_features[mask] = self.mask_token
        
        return masked_features, mask
        
    def _compute_spd_loss(
        self,
        embeddings: torch.Tensor,
        spd_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute shortest path distance loss"""
        num_nodes = embeddings.size(0)
        spd_pred = torch.zeros_like(spd_matrix)
        
        # Compute predicted SPD for all node pairs
        for i in range(num_nodes):
            for j in range(num_nodes):
                spd_pred[i, j] = self.spd_decoder(embeddings[i], embeddings[j])
                
        return F.mse_loss(spd_pred, spd_matrix)
        
    def forward(
        self,
        graph: dgl.DGLGraph,
        features: Dict[str, torch.Tensor],
        spd_matrix: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Average features across modalities
        x = torch.stack(list(features.values())).mean(dim=0)
        
        # Mask features
        masked_x, mask = self._mask_features(x, self.feat_drop_rate)
        
        # Apply MoE
        aligned_x = self.moe(masked_x)
        
        # Apply GNN layers
        h = aligned_x
        for layer in self.gnn_layers:
            h = layer(graph, h).mean(dim=1)
            
        if return_embeddings:
            return h
            
        # Reconstruct features for each domain
        reconstruction_loss = 0
        for domain, decoder in self.domain_decoders.items():
            reconstructed = decoder(h[mask])
            original = features[domain][mask]
            similarity = F.cosine_similarity(reconstructed, original, dim=-1)
            reconstruction_loss += (1 - similarity).pow(self.gamma).mean()
            
        # Compute SPD loss if provided
        spd_loss = 0
        if spd_matrix is not None:
            spd_loss = self._compute_spd_loss(h, spd_matrix)
            
        # Combine losses
        total_loss = reconstruction_loss + self.lambda_spd * spd_loss
        
        return total_loss, h 