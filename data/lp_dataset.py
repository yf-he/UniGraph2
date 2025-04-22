import os
from typing import Optional, Dict, Any, Tuple

import torch
import dgl
import numpy as np
from torch.utils.data import Dataset


class LinkPredictionDataset(Dataset):
    """Dataset for link prediction task"""
    
    def __init__(
        self,
        root: str,
        feat_name: str = "t5vit",
        edge_split_type: str = "time",
        verbose: bool = True,
        device: str = "cpu"
    ):
        """Initialize the dataset
        
        Args:
            root: Root directory containing the dataset
            feat_name: Feature type to use (default: "t5vit")
            edge_split_type: Type of edge split ("time" or "random")
            verbose: Whether to print loading information
            device: Device to load data on ("cpu" or "cuda")
        """
        self.root = root
        self.feat_name = feat_name
        self.edge_split_type = edge_split_type
        self.verbose = verbose
        self.device = device
        
        # Load graph data
        self.graph = self._load_graph()
        
        # Load node features
        self.features = self._load_features()
        
        # Load edge splits
        self.train_edges = self._load_edges("train")
        self.val_edges = self._load_edges("val")
        self.test_edges = self._load_edges("test")
        
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
        
    def _load_edges(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load edge indices for train/val/test split"""
        edge_path = os.path.join(self.root, f"{split}_edges_{self.edge_split_type}.pt")
        if not os.path.exists(edge_path):
            raise FileNotFoundError(f"Edge file not found at {edge_path}")
            
        edges = torch.load(edge_path)
        if self.verbose:
            print(f"Loaded {split} edges with {edges.shape[0]} edges")
            
        return edges.to(self.device)
        
    def __len__(self) -> int:
        return self.graph.num_edges()
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features,
            "train_edges": self.train_edges,
            "val_edges": self.val_edges,
            "test_edges": self.test_edges
        }


class LinkPredictionEvaluator:
    """Evaluator for link prediction task"""
    
    def __init__(self, eval_metric: str = "rocauc"):
        """Initialize the evaluator
        
        Args:
            eval_metric: Evaluation metric ("rocauc" or "hits@k")
        """
        self.eval_metric = eval_metric
        self.expected_input_format = {
            "y_true": "torch.Tensor of shape (num_edges,)",
            "y_pred": "torch.Tensor of shape (num_edges,)",
            "k": "int (only for hits@k metric)"
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
        else:  # hits@k
            k = input_dict.get("k", 10)
            _, indices = torch.topk(y_pred, k)
            metric = (y_true[indices].sum() / k).item()
            
        return {"metric": metric} 