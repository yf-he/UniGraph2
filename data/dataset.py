from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image


class MultimodalGraphDataset(Dataset):
    """Dataset for multimodal graph data."""
    
    def __init__(
        self,
        data_path: str,
        text_model_name: str,
        image_model_name: str,
        max_text_length: int = 512
    ):
        super().__init__()
        self.data_path = data_path
        self.max_text_length = max_text_length
        
        # Initialize tokenizer and image processor
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load data from disk.
        
        Expected format:
        [
            {
                "text": str,
                "image_path": str,
                "graph": {
                    "x": torch.Tensor,  # Node features
                    "edge_index": torch.Tensor,  # Edge indices
                    "edge_attr": torch.Tensor,  # Edge features (optional)
                }
            },
            ...
        ]
        """
        # TODO: Implement data loading logic
        raise NotImplementedError
        
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input."""
        # Tokenize text
        tokens = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }
        
    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process image input."""
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        processed = self.image_processor(image, return_tensors="pt")
        
        return processed["pixel_values"].squeeze(0)
        
    def _process_graph(self, graph_data: Dict) -> Data:
        """Process graph input."""
        # Create PyG Data object
        graph = Data(
            x=graph_data["x"],
            edge_index=graph_data["edge_index"],
            edge_attr=graph_data.get("edge_attr")
        )
        
        return graph
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Data]]:
        """Get a single data item."""
        item = self.data[idx]
        
        # Process each modality
        text_data = self._process_text(item["text"])
        image_data = self._process_image(item["image_path"])
        graph_data = self._process_graph(item["graph"])
        
        return {
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
            "pixel_values": image_data,
            "graph": graph_data
        }
        
        
def collate_fn(
    batch: List[Dict[str, Union[torch.Tensor, Data]]]
) -> Dict[str, Union[torch.Tensor, Batch]]:
    """Collate function for DataLoader."""
    # Separate modalities
    text_data = {
        "input_ids": [],
        "attention_mask": []
    }
    image_data = []
    graph_data = []
    
    for item in batch:
        # Collect text data
        text_data["input_ids"].append(item["input_ids"])
        text_data["attention_mask"].append(item["attention_mask"])
        
        # Collect image data
        image_data.append(item["pixel_values"])
        
        # Collect graph data
        graph_data.append(item["graph"])
    
    # Stack text and image data
    text_batch = {
        k: torch.stack(v) for k, v in text_data.items()
    }
    image_batch = torch.stack(image_data)
    
    # Batch graph data
    graph_batch = Batch.from_data_list(graph_data)
    
    return {
        "input_ids": text_batch["input_ids"],
        "attention_mask": text_batch["attention_mask"],
        "pixel_values": image_batch,
        "graph": graph_batch
    } 