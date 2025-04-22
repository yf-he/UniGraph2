import os
from typing import Optional, Dict, Any

import torch
import dgl
import numpy as np
from torch.utils.data import Dataset


class NodeClassificationDataset(Dataset):
    """Dataset for node classification task"""
    
    def __init__(
        self,
        root: str,
        feat_name: str = "t5vit",
        verbose: bool = True,
        device: str = "cpu"
    ):
        """Initialize the dataset
        
        Args:
            root: Root directory containing the dataset
            feat_name: Feature type to use (default: "t5vit")
            verbose: Whether to print loading information
            device: Device to load data on ("cpu" or "cuda")
        """
        self.root = root
        self.feat_name = feat_name
        self.verbose = verbose
        self.device = device
        
        # Load graph data
        self.graph = self._load_graph()
        
        # Load node features
        self.features = self._load_features()
        
        # Load labels and masks
        self.labels = self._load_labels()
        self.train_mask = self._load_mask("train")
        self.val_mask = self._load_mask("val")
        self.test_mask = self._load_mask("test")
        
    def _load_graph(self) -> dgl.DGLGraph:
        """Load graph structure"""
        graph_path = os.path.join(self.root, "graph.bin")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found at {graph_path}")
            
        graph = dgl.load_graphs(graph_path)[0][0]
        if self.verbose:
            print(f"Loaded graph with {graph.num_nodes()} nodes and {graph.num_edges()} edges")
            
        return graph.to(self.device)
        
    def _load_features(self) -> torch.Tensor:
        """Load node features"""
        feat_path = os.path.join(self.root, f"features_{self.feat_name}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature file not found at {feat_path}")
            
        features = torch.load(feat_path)
        if self.verbose:
            print(f"Loaded features with shape {features.shape}")
            
        return features.to(self.device)
        
    def _load_labels(self) -> torch.Tensor:
        """Load node labels"""
        label_path = os.path.join(self.root, "labels.pt")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found at {label_path}")
            
        labels = torch.load(label_path)
        if self.verbose:
            print(f"Loaded labels with shape {labels.shape}")
            
        return labels.to(self.device)
        
    def _load_mask(self, split: str) -> torch.Tensor:
        """Load node mask for train/val/test split"""
        mask_path = os.path.join(self.root, f"{split}_mask.pt")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found at {mask_path}")
            
        mask = torch.load(mask_path)
        if self.verbose:
            print(f"Loaded {split} mask with {mask.sum()} nodes")
            
        return mask.to(self.device)
        
    def __len__(self) -> int:
        return self.graph.num_nodes()
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "train_mask": self.train_mask[idx],
            "val_mask": self.val_mask[idx],
            "test_mask": self.test_mask[idx]
        }


class NodeClassificationEvaluator:
    """Evaluator for node classification task"""
    
    def __init__(self, eval_metric: str = "rocauc"):
        """Initialize the evaluator
        
        Args:
            eval_metric: Evaluation metric ("rocauc" or "acc")
        """
        self.eval_metric = eval_metric
        self.expected_input_format = {
            "y_true": "torch.Tensor of shape (num_nodes,)",
            "y_pred": "torch.Tensor of shape (num_nodes, num_classes)"
        }
        self.expected_output_format = {
            "metric": "float value between 0 and 1"
        }
        
    def eval(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate predictions
        
        Args:
            input_dict: Dictionary containing predictions and ground truth
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        
        if self.eval_metric == "rocauc":
            from sklearn.metrics import roc_auc_score
            metric = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        else:  # accuracy
            metric = (y_pred.argmax(dim=1) == y_true).float().mean().item()
            
        return {"metric": metric} 