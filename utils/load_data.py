#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Loading Utilities
This module provides utilities for loading and processing graph datasets,
including data loaders for training, evaluation, and in-context learning.
"""

import os
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset, IterableDataset
import dgl
import dgl.dataloading
from datasets import load_large_dataset
from utils.augmentation import create_edge_mask, drop_node_features
from utils.utils import show_occupied_memory

class GraphDataset(Dataset):
    """Base class for graph datasets"""
    
    def __init__(self, 
                 features: Union[torch.Tensor, callable],
                 graph: dgl.DGLGraph,
                 ego_graphs: List[np.ndarray]):
        """Initialize the dataset
        
        Args:
            features: Node features (tensor or callable)
            graph: Input graph
            ego_graphs: List of ego graph node indices
        """
        self.graph = graph
        self.features = features
        self.ego_graphs = ego_graphs
        
    def __len__(self) -> int:
        return len(self.ego_graphs)
        
    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        context_nodes, subgraph = self._get_subgraph(idx)
        if torch.is_tensor(self.features):
            features = self.features[context_nodes].to(torch.float32)
        else:
            features = self.features(context_nodes)
        return subgraph, features
        
    def _get_subgraph(self, idx: int) -> Tuple[np.ndarray, dgl.DGLGraph]:
        context = self.ego_graphs[idx]
        subgraph = self.graph.subgraph(context, store_ids=False)
        return context, subgraph

class GraphDataProvider(Dataset):
    """Provider for multiple graph datasets with weighted sampling"""
    
    def __init__(self,
                 datasets: List[GraphDataset],
                 weights: Optional[List[float]] = None,
                 default_dataset: Optional[str] = None,
                 deepspeed: bool = False,
                 provider_rank: Optional[int] = None,
                 world_size: Optional[int] = None,
                 batch_size: Optional[int] = None):
        """Initialize the data provider
        
        Args:
            datasets: List of graph datasets
            weights: Sampling weights for each dataset
            default_dataset: Default dataset name for length calculation
            deepspeed: Whether using DeepSpeed
            provider_rank: Rank in distributed training
            world_size: World size in distributed training
            batch_size: Batch size for distributed training
        """
        self.deepspeed = deepspeed
        self.provider_rank = provider_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.datasets = datasets
        
        # Calculate dataset lengths and weights
        self.dataset_lengths = np.array([len(d) for d in datasets])
        self.weights = np.array(weights) if weights is not None else self.dataset_lengths
        assert len(self.weights) == len(self.datasets)
        
        # Determine default dataset index
        default_idx = self._get_default_dataset_idx(default_dataset)
        
        # Calculate augmented lengths
        self.augmented_lengths = self._calculate_augmented_lengths(default_idx)
        self.cumulative_lengths = self._calculate_cumulative_lengths()
        
        # Initialize random mappings
        self.random_mappings = self._initialize_random_mappings()
        self.total_shuffle_map = self._initialize_total_shuffle_map()
        
    def _get_default_dataset_idx(self, default_dataset: Optional[str]) -> int:
        """Get index of default dataset"""
        if default_dataset is None:
            return np.argmax(self.dataset_lengths)
        elif default_dataset == "arxiv":
            return np.argmin(self.dataset_lengths)
        elif default_dataset == "products":
            return 1
        elif default_dataset == "papers100M":
            return 2
        elif default_dataset in ["Wiki", "ConceptNet", "FB15K237"]:
            return 3
        else:
            raise ValueError(f"Invalid default dataset: {default_dataset}")
            
    def _calculate_augmented_lengths(self, default_idx: int) -> List[int]:
        """Calculate augmented lengths based on weights"""
        base_length = self.dataset_lengths[default_idx] / self.weights[default_idx]
        return [int(w * base_length) for w in self.weights]
        
    def _calculate_cumulative_lengths(self) -> Dict[int, int]:
        """Calculate cumulative lengths for each dataset"""
        cumsum = np.cumsum(self.augmented_lengths)
        return {i: cumsum[i] for i in range(len(cumsum))}
        
    def _initialize_random_mappings(self) -> List[np.ndarray]:
        """Initialize random mappings for each dataset"""
        return [np.random.permutation(len(d)) for d in self.datasets]
        
    def _initialize_total_shuffle_map(self) -> np.ndarray:
        """Initialize total shuffle map"""
        total_length = sum(self.augmented_lengths)
        shuffle_map = np.arange(total_length)
        np.random.shuffle(shuffle_map)
        return shuffle_map
        
    def __len__(self) -> int:
        return sum(self.augmented_lengths)
        
    def __getitem__(self, index: int) -> Tuple[Tuple[dgl.DGLGraph, torch.Tensor], int]:
        dataset_idx, idx = self._get_dataset_idx(index)
        dataset = self.datasets[dataset_idx]
        idx = self.random_mappings[dataset_idx][idx]
        return dataset[idx], dataset_idx
        
    def _get_dataset_idx(self, index: int) -> Tuple[int, int]:
        """Get dataset index and sample index"""
        if self.deepspeed:
            worker_info = torch.utils.data.get_worker_info()
            worker_start_id = self.provider_rank + self.world_size * worker_info.id * self.batch_size
            if index == worker_start_id:
                np.random.shuffle(self.total_shuffle_map)
                self.random_mappings = self._initialize_random_mappings()
            index = self.total_shuffle_map[index]
            
        for i, length in self.cumulative_lengths.items():
            if index < length:
                dataset_idx = i
                if i == 0:
                    idx = index % self.dataset_lengths[i]
                else:
                    idx = (index - self.cumulative_lengths[dataset_idx - 1]) % self.dataset_lengths[i]
                return dataset_idx, idx
        raise ValueError(f"Index {index} out of bounds")

class Collator(object):
    def __init__(self, drop_edge_rate=0, dataset_drop_edge = [0,1,2], drop_model="random", dataset_drop_feat=[0,1,2], drop_feature_rate_1=0, drop_feature_rate_2=0):
        self._drop_edge_rate = drop_edge_rate
        self.dataset_drop_edge = dataset_drop_edge
        self.drop_model = drop_model
        self.dataset_drop_feat = dataset_drop_feat
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2

    def drop_edge(self, g, dataset_id=None):
        if self._drop_edge_rate <= 0:
            return g

        if self.drop_model == "random" or dataset_id==1 :
            g = g.remove_self_loop()
            mask_index1 = mask_edge(g, self._drop_edge_rate)
            g1 = dgl.remove_edges(g, mask_index1.to(torch.int32)).add_self_loop()
            return g1
        elif self.drop_model == "directed_to_undirected":
            random_number = random.random()
            if random_number <self._drop_edge_rate:
                g = dgl.to_bidirected(g) 
                g = g.remove_self_loop().add_self_loop()
            return g
        else:
            raise ValueError

    def __call__(self, batch):
        #[0][0]subg, [0][1]context_id，[1] dataset_id 
        subgraphs = [x[0][0] for x in batch]
        targets = np.cumsum([0] + [x.num_nodes() for x in subgraphs])[:-1]
        targets = torch.from_numpy(targets)
        feats = torch.cat([x[0][1] for x in batch], dim=0)

        context_nodes_dataset_id = [x[1] for x in batch]
        subg = dgl.batch(subgraphs)

        if self._drop_edge_rate > 0:
            drop_subgraphs = []
            for idx, dataset_id in enumerate(context_nodes_dataset_id):
                if dataset_id in self.dataset_drop_edge:
                    drop_g = self.drop_edge(subgraphs[idx],dataset_id=dataset_id)
                    drop_subgraphs.append(drop_g)
                else:
                    drop_subgraphs.append(subgraphs[idx])

            drop_subg = dgl.batch(drop_subgraphs)
            return subg, targets, feats, context_nodes_dataset_id, drop_subg
        else:
            return subg, targets, feats, context_nodes_dataset_id

def load_dataloader(load_type: str,
                   dataset_name: str,
                   args: Any,
                   pretrain_seed: Optional[int] = None) -> Union[DataLoader, Tuple[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], DataLoader]]:
    """Load appropriate dataloader based on type
    
    Args:
        load_type: Type of dataloader ("eval" or "train")
        dataset_name: Name of the dataset
        args: Training arguments
        pretrain_seed: Random seed for pretraining
        
    Returns:
        Dataloader or tuple of (split sizes, labels, dataloader)
    """
    if hasattr(args, "task") and args.task in ["node_classification", "link_prediction"]:
        from utils.mm_graph_benchmark import setup_mm_graph_benchmark
        loader, evaluator = setup_mm_graph_benchmark(args)
        
        if args.task == "node_classification":
            graph = loader.graph
            train_mask = graph.ndata["train_mask"]
            val_mask = graph.ndata["val_mask"]
            test_mask = graph.ndata["test_mask"]
            
            num_train = train_mask.sum().item()
            num_val = val_mask.sum().item()
            num_test = test_mask.sum().item()
            
            labels = graph.ndata["label"]
            train_labels = labels[train_mask]
            val_labels = labels[val_mask]
            test_labels = labels[test_mask]
            
            return (num_train, num_val, num_test), (train_labels, val_labels, test_labels), None
        else:
            edge_split = loader.edge_split
            return None, None, None
    elif load_type == "eval":
        return _load_eval_dataloader(dataset_name, args)
    else:
        return _load_train_dataloader(args, pretrain_seed)

def _load_eval_dataloader(dataset_name: str, args: Any) -> Tuple[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor], DataLoader]:
    """Load evaluation dataloader"""
    feats, graph, labels, split_idx, ego_graph_nodes = load_large_dataset(
        dataset_name,
        args.data_dir,
        args.ego_graph_file_path,
        args.no_scale,
        args.multi_scale,
        feat_type=args.feat_type,
        drop_model=args.drop_model
    )
    
    if ego_graph_nodes is not None:
        ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
    else:
        ego_graph_nodes = [np.array([0, 1, 1, 1])]
        
    dataloader = LocalClusteringLoader(
        root_nodes=ego_graph_nodes,
        graph=graph,
        feats=feats,
        batch_size=args.batch_size_f,
        shuffle=False,
        drop_last=False,
        persistent_workers=True,
        num_workers=args.prob_num_workers
    )
    
    num_train, num_val, num_test = [split_idx[k].shape[0] for k in ["train", "valid", "test"]]
    train_idx = np.arange(0, num_train)
    val_idx = np.arange(num_train, num_train + num_val)
    test_idx = np.arange(num_train + num_val, num_train + num_val + num_test)
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    
    return (num_train, num_val, num_test), (train_labels, val_labels, test_labels), dataloader

def _load_train_dataloader(args: Any, pretrain_seed: Optional[int] = None) -> DataLoader:
    """Load training dataloader"""
    pretrain_datasets = args.pretrain_dataset or ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]
    datasets = []
    
    for dataset_name in pretrain_datasets:
        feats, graph, _, _, ego_graph_nodes = load_large_dataset(
            dataset_name,
            args.data_dir,
            None,
            args.no_scale,
            args.multi_scale,
            feat_type=args.feat_type,
            drop_model=args.drop_model
        )
        
        if dataset_name in ["Wiki", "ConceptNet", "FB15K237"]:
            ego_graph_nodes = ego_graph_nodes[0]
        else:
            ego_graph_nodes = ego_graph_nodes[0] + ego_graph_nodes[1] + ego_graph_nodes[2]
            
        datasets.append(GraphDataset(feats, graph, ego_graph_nodes))
        
    if args.deepspeed:
        provider_rank = torch.distributed.get_rank()
        world_size = torch.cuda.device_count()
    else:
        provider_rank = None
        world_size = None
        
    data_provider = GraphDataProvider(
        datasets,
        weights=args.weight,
        default_dataset=args.default_dataset,
        deepspeed=args.deepspeed,
        provider_rank=provider_rank,
        world_size=world_size,
        batch_size=args.batch_size
    )
    
    if args.deepspeed:
        world_size = torch.cuda.device_count()
        rank = torch.distributed.get_rank()
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_provider,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        def worker_init_fn(worker_id: int) -> None:
            assert pretrain_seed is not None
            np.random.seed(pretrain_seed)
            
        dataloader = DataLoader(
            data_provider,
            collate_fn=Collator(
                drop_edge_rate=args.drop_edge_rate,
                dataset_drop_edge=[pretrain_datasets.index(name) for name in args.dataset_drop_edge] if args.drop_edge_rate > 0 else None,
                drop_model=args.model,
                dataset_drop_feat=[pretrain_datasets.index(name) for name in args.dataset_drop_feat] if args.dataset_drop_feat is not None else list(range(len(pretrain_datasets))),
                drop_feature_rate_1=args.drop_feature_rate_1,
                drop_feature_rate_2=args.drop_feature_rate_2
            ),
            batch_size=args.batch_size,
            num_workers=args.pretrain_num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            data_provider,
            collate_fn=Collator(
                drop_edge_rate=args.drop_edge_rate,
                dataset_drop_edge=[pretrain_datasets.index(name) for name in args.dataset_drop_edge] if args.drop_edge_rate > 0 else None,
                drop_model=args.model,
                dataset_drop_feat=[pretrain_datasets.index(name) for name in args.dataset_drop_feat] if args.dataset_drop_feat is not None else list(range(len(pretrain_datasets))),
                drop_feature_rate_1=args.drop_feature_rate_1,
                drop_feature_rate_2=args.drop_feature_rate_2
            ),
            batch_size=args.batch_size,
            num_workers=args.pretrain_num_workers,
            shuffle=True,
            persistent_workers=True,
            drop_last=False
        )
        
    return dataloader

class LocalClusteringLoader(DataLoader):
    def __init__(self, root_nodes, graph, feats, **kwargs):
        self.graph = graph
        self.ego_graph_nodes = root_nodes
        self.feats = feats
        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [self.graph.subgraph(ego_nodes[i], store_ids=False) for i in range(len(ego_nodes))]
        sg = dgl.batch(subgs)
        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]
        if torch.is_tensor(self.feats):
            sg.ndata["feat"] = self.feats[nodes].to(torch.float32)
        else:
            sg.ndata["feat"] = self.feats(nodes.numpy()) 
        targets = torch.from_numpy(cum_num_nodes)      
        return sg, targets, None, nodes

class LinearProbingDataLoader(DataLoader):
    def __init__(self, idx, feats, labels=None, **kwargs):
        self.labels = labels
        self.feats = feats

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=idx, **kwargs)

    def __collate_fn__(self, batch_idx):
        feats = self.feats[batch_idx]
        label = self.labels[batch_idx]
        return feats, label

class BaseDataset(object):
    dataset_name: str = "basic_dataset"
    def __init__(
            self,  
            feat, 
            graph,
            ego_graph,
        ):
        #print(f"--- loading {self.dataset_name} ---")

        self.graph = graph
        self.feats = feat
        self.ego_graphs = ego_graph

    def __getitem__(self, idx) :
        context_nodes, subg  = self.get_subgraph(idx)
        if torch.is_tensor(self.feats):
            feats = self.feats[context_nodes].to(torch.float32)
        else:
            feats = self.feats(context_nodes)   
        return subg, feats
        
    def __len__(self):
        return len(self.ego_graphs)
    
    def get_subgraph(self, idx):
        context = self.ego_graphs[idx]
        subg = self.graph.subgraph(context, store_ids=False)
        return context, subg
       
class DataProvider(object):
    def __init__(self, datasets, weights=None, default_dataset=None, deepspeed=False, provider_rank=None, world_size=None,batch_size=None): #weight:list
        self.deepspeed = deepspeed
        self.provider_rank = provider_rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.datasets = datasets
        self.len_dataset = np.array([len(i) for i in datasets])
        if weights is None:
            self.weights = np.array(self.len_dataset)
        else:
            self.weights = np.array(weights)
        assert len(self.weights) == len(self.datasets)
        if default_dataset==None:
            maxlen_idx = np.argmax(self.len_dataset)
        elif default_dataset=="arxiv":
            maxlen_idx = np.argmin(self.len_dataset)
        elif default_dataset=="products":
            maxlen_idx = 1
        elif default_dataset=="papers100M":
            maxlen_idx = 2
        elif default_dataset=="Wiki" or default_dataset=="ConceptNet" or default_dataset=="FB15K237":
            maxlen_idx = 3
        else:
            print("default_dataset point error")
            exit(0)
        num_one_weight = self.len_dataset[maxlen_idx]/self.weights[maxlen_idx]
        self.augement_len = []
        for i in range(len(self.datasets)):
            self.augement_len.append(int(self.weights[i]*num_one_weight))
        self.length_dataset = np.cumsum([j for j in self.augement_len])
        self.length_dataset = {
            i: self.length_dataset[i] for i in range(len(self.length_dataset))
        }
        self.random_mapping = self.bulid_each_dataset_random_map()
        
        total_num_for_train = sum(self.augement_len)
        self.tot_shuffle_map = np.arange(total_num_for_train)
        np.random.shuffle(self.tot_shuffle_map)
    
    def bulid_each_dataset_random_map(self):
        random_mapping = [] 
        for l in self.len_dataset:
            mapping = np.arange(l)
            np.random.shuffle(mapping)
            random_mapping.append(mapping)
        return random_mapping
        
    def __len__(self):
        lens = self.augement_len 
        return sum(lens)

    def __getitem__(self, index):
        dataset_idx, idx = self.get_dataset_idx(index)
        dataset = self.datasets[dataset_idx]
        # remapping makes index random 
        idx = self.random_mapping[dataset_idx][idx]
        return dataset[idx] , dataset_idx

    def get_dataset_idx(self, index):
        if self.deepspeed:
            worker_start_id = self.provider_rank + self.world_size* torch.utils.data.get_worker_info().id * self.batch_size 
            if index == worker_start_id :  
                #print(f"rank {self.provider_rank} worker id {torch.utils.data.get_worker_info().id} shuffle data provider. start_id {worker_start_id}") 
                np.random.shuffle(self.tot_shuffle_map) #shuffle total map
                self.random_mapping = self.bulid_each_dataset_random_map()  #shuffle seqerate map
                #print("rank",self.provider_rank,"workerid",torch.utils.data.get_worker_info(),self.tot_shuffle_map,self.random_mapping)  
            index = self.tot_shuffle_map[index]
        #print("rank",self.provider_rank,"workerid",torch.utils.data.get_worker_info(),self.tot_shuffle_map,self.random_mapping) 
        
        for i, x in self.length_dataset.items():
            if index < x:
                dataset_idx = i
                if i==0:
                    idx = index % self.len_dataset[i]
                else:
                    idx = (index - self.length_dataset[dataset_idx - 1])% self.len_dataset[i]
                return dataset_idx, idx
        raise ValueError
    
class InContextDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, args, eval_tasks=None, num_label=None, num_support=None, num_query=None):
        super(InContextDataset, self).__init__()
        self.args = args
        self.fs_label = args.fs_label
        self.sample_position = args.sample_position
        self.device = args.device
        self.dataset_name = dataset_name
        self.node_classify_task = ["ogbn-arxiv","Cora","Pubmed"]
        self.link_predict_task = ["FB15K237","WN18RR"]
        self.num_label = self.args.num_label if num_label is None else num_label
        self.num_support = self.args.num_support if num_support is None else num_support
        self.num_query = self.args.num_query if num_query is None else num_query

        feats, self.graph, self.labels, self.split_idx, ego_graph_nodes = load_large_dataset(dataset_name, args.data_dir,
                                                                                            args.ego_graph_file_path,  args.no_scale, args.multi_scale, feat_type=args.feat_type, drop_model="random")

        self.split_len = {"train": range(self.split_idx["train"].shape[0]),
                         "valid": range(self.split_idx["train"].shape[0],self.split_idx["train"].shape[0]+self.split_idx["valid"].shape[0]), 
                         "test": range(self.split_idx["train"].shape[0]+self.split_idx["valid"].shape[0],self.split_idx["train"].shape[0]+self.split_idx["valid"].shape[0]+self.split_idx["test"].shape[0])}

        self.split_num_start = {"train": 0,
                                "valid": self.split_idx["train"].shape[0], 
                                "test":self.split_idx["train"].shape[0]+self.split_idx["valid"].shape[0]}
       
        self.graph.ndata["feat"] = feats
        print(f"Finish loading the graph")
        self.label_dict = dict()
        for split in ["train", "valid", "test"]:
            for idx in self.split_len[split]:
                label = self.labels[idx].item()
                if not np.isnan(label):
                    if label not in self.label_dict:
                        self.label_dict[label] = {"train": [], "valid": [], "test": []}
                    self.label_dict[label][split].append(self.split_idx[split][idx-self.split_num_start[split]])
        label_dict_tmp = {}
        for k,v in self.label_dict.items():
            v["total"] = v["train"]+v["valid"]+v["test"]
            label_dict_tmp[k] = v
        self.label_dict = label_dict_tmp
        self.total_labels = len(self.label_dict.keys())
        self.total_nodes = self.graph.num_nodes()
        if eval_tasks is None:
            self.total_steps = args.total_steps
        else:
            self.total_steps = eval_tasks
        
    def get_khop_graph(self, nid, khop=2, drop_node=False):
        graph, _ = dgl.khop_out_subgraph(self.graph, nid.to(torch.int32), khop)
        return graph
    
    def generate_batch(self, batch_type="mt"):
        def sample(sample_list, size):
            if len(sample_list) >= size:
                select_idx = np.random.choice(len(sample_list), size=size, replace=False)
            else:
                select_idx = np.random.choice(len(sample_list), size=size, replace=True)
            return [sample_list[i] for i in select_idx]

        m = self.num_label
        if batch_type == "mt":
            if self.fs_label == "total":
                current_labels = np.random.choice(range(self.total_labels), m, replace=False) 
            elif self.fs_label == "ofa":
                if self.dataset_name == "ogbn-arxiv":
                    current_labels = np.random.choice( [17,39,10,5,16,15,18,37,30,33] , m, replace=False)
                elif self.dataset_name == "FB15K237":
                    current_labels = np.random.choice( [97, 218, 227, 86, 217, 39, 202, 87, 221, 178, 40, 194, 1, 71, 150, 114, 56, 107, 224, 179, 166, 183, 50, 143, 234, 154, 129, 59, 55, 23, 7, 8, 108, 151, 22, 139, 233, 173, 26, 188, 35, 57, 62, 70, 189, 6, 28, 163] , m, replace=False)
                else:
                    raise ValueError
            else:
                raise ValueError

            while True:
                flag = True
                for label in current_labels:
                    if len(self.label_dict[label]["train"]) == 0 or len(self.label_dict[label]["test"]) == 0:
                        flag = False
                if flag:
                    break
                else:
                    current_labels = np.random.choice(range(self.total_labels), m, replace=False)
        else:
            raise ValueError(batch_type)
        support_examples = []
        query_examples = []
        support_labels = []
        query_labels = []

        k = self.num_support
        n = self.num_query
        for idx, label in enumerate(current_labels):
            if batch_type == "mt":
                if self.sample_position == "train_test" :
                    examples = sample(self.label_dict[label]["train"], k) + sample(self.label_dict[label]["test"], n)
                elif self.sample_position == "total" :
                    examples = sample(self.label_dict[label]["total"], k+n) 

            for i in range(k):
                if self.dataset_name in self.node_classify_task: #example: [nodeid,]
                    support_examples.append(self.get_khop_graph(examples[i], self.args.khop, drop_node=True))
                elif  self.dataset_name in self.link_predict_task: #example: [tensor(nodeid1,nodeid2),]
                    support_examples.append(self.get_khop_graph(examples[i][0], self.args.khop, drop_node=True))
                    support_examples.append(self.get_khop_graph(examples[i][1], self.args.khop, drop_node=True))
                support_labels.append(idx)
                
            for i in range(n):
                if self.dataset_name in self.node_classify_task: #example: [nodeid,]
                    query_examples.append(self.get_khop_graph(examples[k + i], self.args.khop, drop_node=True))
                elif  self.dataset_name in self.link_predict_task: #example: [tensor(nodeid1,nodeid2),]
                    query_examples.append(self.get_khop_graph(examples[k + i][0], self.args.khop, drop_node=True))
                    query_examples.append(self.get_khop_graph(examples[k + i][1], self.args.khop, drop_node=True))
                query_labels.append(idx)    
        all_examples = support_examples + query_examples
        data_graph = dgl.batch(all_examples)
        labels = torch.LongTensor(query_labels)
        chosen_labels = torch.LongTensor(current_labels)
        batch = {
            "data_graph": data_graph,
            "labels": labels,
            "chosen_labels": chosen_labels,
        }
        return batch


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = max(worker_info.num_workers, 1)
        for _ in range(self.total_steps // num_workers):
            mt_batch = self.generate_batch(batch_type="mt")

            yield mt_batch

    def __len__(self):
        return self.total_steps

def setup_incontext_dataloader(dataset_name, args):
    eval_dataset = InContextDataset(dataset_name, args, eval_tasks=args.total_steps, num_label=args.eval_num_label, num_support=args.eval_num_support, num_query=args.eval_num_query)
    dataloader = DataLoader(eval_dataset, batch_size=None, num_workers=1)
    return dataloader

