from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class UniGraph2Trainer(pl.LightningModule):
    """Trainer module for UniGraph2."""
    
    def __init__(self, model: nn.Module, config: Dict):
        super().__init__()
        self.model = model
        self.config = config
        
        # Get loss weights
        self.loss_weights = config["training"]["loss_weights"]
        
        # Temperature parameter for contrastive loss
        self.temperature = 0.07
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get optimizer config
        opt_config = self.config["training"]["optimizer"]
        
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            betas=(opt_config["beta1"], opt_config["beta2"]),
            eps=opt_config["eps"]
        )
        
        # Create scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
        
    def _compute_contrastive_loss(
        self,
        embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute contrastive loss between different modalities."""
        loss = 0
        num_pairs = 0
        
        # Get all modality pairs
        modalities = list(embeddings.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                
                # Get embeddings
                emb1 = F.normalize(embeddings[mod1], dim=-1)
                emb2 = F.normalize(embeddings[mod2], dim=-1)
                
                # Compute similarity matrix
                sim_matrix = torch.matmul(emb1, emb2.T) / self.temperature
                
                # Labels are on the diagonal
                labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                
                # Compute loss in both directions
                loss_i2j = F.cross_entropy(sim_matrix, labels)
                loss_j2i = F.cross_entropy(sim_matrix.T, labels)
                
                loss += (loss_i2j + loss_j2i) / 2
                num_pairs += 1
                
        return loss / max(num_pairs, 1)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        # Forward pass
        embeddings = self.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            graph=batch.get("graph")
        )
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(embeddings)
        loss = self.loss_weights["contrastive"] * contrastive_loss
        
        # Log losses
        self.log("train/contrastive_loss", contrastive_loss)
        self.log("train/total_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Forward pass
        embeddings = self.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            graph=batch.get("graph")
        )
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(embeddings)
        loss = self.loss_weights["contrastive"] * contrastive_loss
        
        # Log losses
        self.log("val/contrastive_loss", contrastive_loss)
        self.log("val/total_loss", loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        # Forward pass
        embeddings = self.model(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            graph=batch.get("graph")
        )
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(embeddings)
        loss = self.loss_weights["contrastive"] * contrastive_loss
        
        # Log losses
        self.log("test/contrastive_loss", contrastive_loss)
        self.log("test/total_loss", loss)
        
        return loss 