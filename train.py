import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping
)
from pytorch_lightning.loggers import WandbLogger

from data.datamodule import MultimodalGraphDataModule
from models.unigraph2_model import UniGraph2Model
from trainers.trainer import UniGraph2Trainer
from models.unigraph2 import UniGraph2
from data.lp_dataset import LinkPredictionDataset
from data.nc_dataset import NodeClassificationDataset


def compute_spd_matrix(graph: dgl.DGLGraph) -> torch.Tensor:
    """Compute shortest path distance matrix for a graph"""
    num_nodes = graph.num_nodes()
    spd_matrix = torch.full((num_nodes, num_nodes), float('inf'))
    
    # Convert graph to adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    src, dst = graph.edges()
    adj_matrix[src, dst] = 1
    adj_matrix[dst, src] = 1  # Undirected graph
    
    # Initialize SPD matrix with direct connections
    spd_matrix[adj_matrix == 1] = 1
    np.fill_diagonal(spd_matrix, 0)
    
    # Floyd-Warshall algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if spd_matrix[i, k] + spd_matrix[k, j] < spd_matrix[i, j]:
                    spd_matrix[i, j] = spd_matrix[i, k] + spd_matrix[k, j]
                    
    return spd_matrix


class UniGraph2Trainer(pl.LightningModule):
    """PyTorch Lightning trainer for UniGraph2"""
    
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        
    def forward(self, batch: Dict) -> torch.Tensor:
        graph = batch["graph"]
        features = batch["features"]
        spd_matrix = compute_spd_matrix(graph).to(self.device)
        
        loss, _ = self.model(graph, features, spd_matrix)
        return loss
        
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss
        
    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    input_dims = {
        "text": 768,  # T5-ViT features
        "image": 512  # CLIP features
    }
    
    model = UniGraph2(
        input_dims=input_dims,
        hidden_dim=768,
        num_experts=8,
        num_selected_experts=2,
        num_layers=3
    ).to(device)
    
    # Create data module
    datamodule = MultimodalGraphDataModule(
        data_dir="data/example",
        feat_name="t5vit",
        edge_split_type="time",
        batch_size=32,
        num_workers=4,
        device=device
    )
    
    # Create trainer
    trainer = UniGraph2Trainer(model)
    
    # Create logger
    logger = WandbLogger(
        project="unigraph2",
        name="multimodal_graph_learning"
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min"
        )
    ]
    
    # Create PyTorch Lightning trainer
    pl_trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0
    )
    
    # Train model
    pl_trainer.fit(trainer, datamodule=datamodule)
    
    # Test model
    pl_trainer.test(trainer, datamodule=datamodule)


if __name__ == "__main__":
    main() 