#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MM-Graph Benchmark Utilities
This module provides utilities for loading and evaluating datasets from the mm-graph-benchmark,
including both node classification and link prediction tasks.
"""

import os
from typing import Dict, Any, Optional, Union, Tuple
import torch
import dgl
from nc_dataset import NodeClassificationDataset, NodeClassificationEvaluator
from lp_dataset import LinkPredictionDataset, LinkPredictionEvaluator

class MMGraphLoader:
    """Loader for MM-Graph Benchmark datasets"""
    
    def __init__(self,
                 data_path: str,
                 dataset_name: str,
                 feat_name: str = "t5vit",
                 edge_split_type: Optional[str] = None,
                 verbose: bool = True,
                 device: str = "cpu"):
        """Initialize the loader
        
        Args:
            data_path: Path to the dataset directory
            dataset_name: Name of the dataset
            feat_name: Feature type to use (default: "t5vit")
            edge_split_type: Type of edge split for link prediction (optional)
            verbose: Whether to print loading information
            device: Device to load data on ("cpu" or "cuda")
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.feat_name = feat_name
        self.edge_split_type = edge_split_type
        self.verbose = verbose
        self.device = device
        self.dataset = None
        self.graph = None
        self.edge_split = None
        
    def load_node_classification(self) -> dgl.DGLGraph:
        """Load dataset for node classification task
        
        Returns:
            DGLGraph with node features, labels, and masks
        """
        self.dataset = NodeClassificationDataset(
            root=os.path.join(self.data_path, self.dataset_name),
            feat_name=self.feat_name,
            verbose=self.verbose,
            device=self.device
        )
        self.graph = self.dataset.graph
        return self.graph
        
    def load_link_prediction(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load dataset for link prediction task
        
        Returns:
            Dictionary containing edge splits for train/valid/test
        """
        self.dataset = LinkPredictionDataset(
            root=os.path.join(self.data_path, self.dataset_name),
            feat_name=self.feat_name,
            edge_split_type=self.edge_split_type,
            verbose=self.verbose,
            device=self.device
        )
        self.graph = self.dataset.graph
        self.edge_split = self.dataset.get_edge_split()
        return self.edge_split

class MMGraphEvaluator:
    """Evaluator for MM-Graph Benchmark tasks"""
    
    def __init__(self,
                 task: str = "node_classification",
                 eval_metric: Optional[str] = None):
        """Initialize the evaluator
        
        Args:
            task: Task type ("node_classification" or "link_prediction")
            eval_metric: Evaluation metric for node classification ("rocauc" or "acc")
        """
        self.task = task
        if task == "node_classification":
            self.evaluator = NodeClassificationEvaluator(eval_metric=eval_metric)
        else:
            self.evaluator = LinkPredictionEvaluator()
            
    def evaluate(self, input_dict: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate predictions
        
        Args:
            input_dict: Dictionary containing predictions and ground truth
            
        Returns:
            Dictionary of evaluation metrics
        """
        return self.evaluator.eval(input_dict=input_dict)
        
    @property
    def input_format(self) -> Dict[str, Any]:
        """Get expected input format for evaluation"""
        return self.evaluator.expected_input_format
        
    @property
    def output_format(self) -> Dict[str, Any]:
        """Get expected output format for evaluation"""
        return self.evaluator.expected_output_format

def setup_mm_graph_benchmark(args: Any) -> Union[MMGraphLoader, Tuple[MMGraphLoader, MMGraphEvaluator]]:
    """Setup MM-Graph Benchmark for training or evaluation
    
    Args:
        args: Training arguments containing dataset configuration
        
    Returns:
        MMGraphLoader or tuple of (MMGraphLoader, MMGraphEvaluator)
    """
    loader = MMGraphLoader(
        data_path=args.data_path,
        dataset_name=args.dataset,
        feat_name=args.feat_name,
        edge_split_type=args.edge_split_type if hasattr(args, "edge_split_type") else None,
        verbose=not args.no_verbose,
        device=args.device
    )
    
    if args.task == "node_classification":
        loader.load_node_classification()
        evaluator = MMGraphEvaluator(
            task="node_classification",
            eval_metric=args.eval_metric if hasattr(args, "eval_metric") else "rocauc"
        )
    else:
        loader.load_link_prediction()
        evaluator = MMGraphEvaluator(task="link_prediction")
        
    return loader, evaluator 