#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Augmentation Utilities
This module provides utilities for augmenting graph data through feature masking
and edge dropping techniques used in UniGraph2.
"""

from typing import Tuple
import torch
import numpy as np
import dgl

def create_random_augmentation(graph: dgl.DGLGraph,
                             features: torch.Tensor,
                             feat_drop_rate: float,
                             edge_mask_rate: float) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Create a randomly augmented graph and features
    
    Args:
        graph: Input graph
        features: Node features
        feat_drop_rate: Feature dropout rate
        edge_mask_rate: Edge masking rate
        
    Returns:
        Tuple of (augmented graph, augmented features)
    """
    # Create new graph with same number of nodes
    new_graph = dgl.graph([]).to(graph.device)
    new_graph.add_nodes(graph.number_of_nodes())
    
    # Mask edges
    edge_mask = create_edge_mask(graph, edge_mask_rate)
    src, dst = graph.edges()
    
    # Add masked edges to new graph
    new_src = src[edge_mask].to(torch.int64)
    new_dst = dst[edge_mask].to(torch.int64)
    new_graph.add_edges(new_src, new_dst)
    
    # Drop features
    new_features = drop_node_features(features, feat_drop_rate)
    
    return new_graph, new_features

def drop_node_features(features: torch.Tensor,
                      drop_prob: float) -> torch.Tensor:
    """Randomly drop node features
    
    Args:
        features: Node features to augment
        drop_prob: Dropout probability
        
    Returns:
        Augmented features
    """
    # Create dropout mask
    drop_mask = torch.empty(
        (features.size(1),),
        dtype=torch.float32,
        device=features.device
    ).uniform_(0, 1) < drop_prob
    
    # Apply dropout
    features = features.clone()
    features[:, drop_mask] = 0
    
    return features

def create_edge_mask(graph: dgl.DGLGraph,
                    mask_prob: float) -> torch.Tensor:
    """Create edge mask for graph
    
    Args:
        graph: Input graph
        mask_prob: Masking probability
        
    Returns:
        Boolean mask tensor for edges
    """
    num_edges = graph.number_of_edges()
    mask_rates = torch.FloatTensor(np.ones(num_edges) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
