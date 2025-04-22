from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, global_mean_pool


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with multi-head attention."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        residual: bool = True
    ):
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim)
            else:
                self.res_fc = nn.Identity()
                
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GAT layer."""
        # Apply GAT
        out = self.gat(x, edge_index)
        
        # Apply residual connection if enabled
        if self.residual:
            out = out + self.res_fc(x)
            
        # Apply dropout and layer norm
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        return out


class GraphEncoder(nn.Module):
    """Graph encoder using Graph Attention Networks."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Get configuration
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        dropout = config["dropout"]
        
        # Input embedding layer
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Stack of GAT layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass through the graph encoder."""
        # Get graph components
        x, edge_index = graph.x, graph.edge_index
        batch = graph.batch if isinstance(graph, Batch) else None
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply GAT layers
        for layer in self.layers:
            x = layer(x, edge_index)
            
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
            
        # Final projection
        x = self.output_proj(x)
        
        return x 