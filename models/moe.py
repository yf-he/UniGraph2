from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert in the Mixture of Experts."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts for feature alignment."""
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        noisy_gating: bool = True,
        noise_epsilon: float = 1e-2
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon
        
        # Create experts
        self.experts = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
    def _compute_gating_scores(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating scores for each expert."""
        # x shape: (batch_size, num_modalities, input_dim)
        batch_size, num_modalities, _ = x.shape
        
        # Reshape for gating
        x_flat = x.view(-1, self.input_dim)  # (batch_size * num_modalities, input_dim)
        
        # Get raw gating scores
        gates = self.gate(x_flat)  # (batch_size * num_modalities, num_experts)
        
        if self.noisy_gating and self.training:
            # Add noise for exploration during training
            noise = torch.randn_like(gates) * self.noise_epsilon
            gates = gates + noise
        
        # Get top-k experts for each input
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_gates, dim=-1)
        
        # Create mask for selected experts
        mask = torch.zeros_like(gates).scatter_(-1, top_k_indices, top_k_gates)
        
        return mask, top_k_indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE."""
        batch_size, num_modalities, input_dim = x.shape
        
        # Compute gating scores
        gates, top_k_indices = self._compute_gating_scores(x)
        
        # Reshape inputs for parallel processing
        x_flat = x.view(-1, input_dim)  # (batch_size * num_modalities, input_dim)
        
        # Initialize output tensor
        final_output = torch.zeros_like(x_flat)
        
        # Compute capacity per expert
        capacity = int(self.capacity_factor * (batch_size * num_modalities) / self.num_experts)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get indices of inputs routed to this expert
            expert_mask = gates[:, i] > 0
            if not expert_mask.any():
                continue
                
            # Get expert inputs and their gates
            expert_inputs = x_flat[expert_mask]
            expert_gates = gates[expert_mask, i].unsqueeze(-1)
            
            # Handle capacity overflow
            if expert_inputs.size(0) > capacity:
                # Randomly drop examples that exceed capacity
                perm = torch.randperm(expert_inputs.size(0))
                expert_inputs = expert_inputs[perm[:capacity]]
                expert_gates = expert_gates[perm[:capacity]]
            
            # Process inputs through expert
            expert_output = expert(expert_inputs)
            
            # Weight outputs by gates and add to final output
            final_output[expert_mask] += expert_output * expert_gates
            
        # Reshape output back to original dimensions
        output = final_output.view(batch_size, num_modalities, input_dim)
        
        return output 