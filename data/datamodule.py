import os
from typing import Dict, Optional, Tuple

import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class MultimodalGraphDataset(Dataset):
    """Dataset for multimodal graph data"""
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        feat_name: str = "t5vit",
        edge_split_type: str = "time",
        device: str = "cpu"
    ):
        """Initialize the dataset
        
        Args:
            root: Root directory containing the dataset
            split: Data split ("train", "val", or "test")
            feat_name: Feature type to use (default: "t5vit")
            edge_split_type: Type of edge split ("time" or "random")
            device: Device to load data on ("cpu" or "cuda")
        """
        self.root = root
        self.split = split
        self.feat_name = feat_name
        self.edge_split_type = edge_split_type
        self.device = device
        
        # Load graph data
        self.graph = self._load_graph()
        
        # Load features for each modality
        self.features = self._load_features()
        
        # Load masks and labels if they exist
        self.mask = self._load_mask()
        self.labels = self._load_labels()
        
    def _load_graph(self) -> dgl.DGLGraph:
        """Load graph structure"""
        graph_path = os.path.join(self.root, "graph.bin")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at {graph_path}")
            
        graph = dgl.load_graphs(graph_path)[0][0]
        return graph.to(self.device)
        
    def _load_features(self) -> Dict[str, torch.Tensor]:
        """Load features for each modality"""
        features = {}
        
        # Load text features
        text_path = os.path.join(self.root, f"features_{self.feat_name}.pt")
        if os.path.exists(text_path):
            features["text"] = torch.load(text_path).to(self.device)
            
        # Load image features
        image_path = os.path.join(self.root, "features_clip.pt")
        if os.path.exists(image_path):
            features["image"] = torch.load(image_path).to(self.device)
            
        return features
        
    def _load_mask(self) -> Optional[torch.Tensor]:
        """Load node mask for the split"""
        mask_path = os.path.join(self.root, f"{self.split}_mask.pt")
        if os.path.exists(mask_path):
            return torch.load(mask_path).to(self.device)
        return None
        
    def _load_labels(self) -> Optional[torch.Tensor]:
        """Load node labels if they exist"""
        label_path = os.path.join(self.root, "labels.pt")
        if os.path.exists(label_path):
            return torch.load(label_path).to(self.device)
        return None
        
    def __len__(self) -> int:
        return self.graph.num_nodes()
        
    def __getitem__(self, idx: int) -> Dict:
        item = {
            "graph": self.graph,
            "features": self.features
        }
        
        if self.mask is not None:
            item["mask"] = self.mask[idx]
            
        if self.labels is not None:
            item["label"] = self.labels[idx]
            
        return item


class MultimodalGraphDataModule(LightningDataModule):
    """Data module for multimodal graph data"""
    
    def __init__(
        self,
        data_dir: str,
        feat_name: str = "t5vit",
        edge_split_type: str = "time",
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cpu"
    ):
        super().__init__()
        self.data_dir = data_dir
        self.feat_name = feat_name
        self.edge_split_type = edge_split_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each split"""
        if stage == "fit" or stage is None:
            self.train_dataset = MultimodalGraphDataset(
                root=self.data_dir,
                split="train",
                feat_name=self.feat_name,
                edge_split_type=self.edge_split_type,
                device=self.device
            )
            self.val_dataset = MultimodalGraphDataset(
                root=self.data_dir,
                split="val",
                feat_name=self.feat_name,
                edge_split_type=self.edge_split_type,
                device=self.device
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = MultimodalGraphDataset(
                root=self.data_dir,
                split="test",
                feat_name=self.feat_name,
                edge_split_type=self.edge_split_type,
                device=self.device
            )
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
        
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate function for batching"""
        # Since we're working with a single graph, we just return the first item
        return batch[0] 