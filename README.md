# UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs

UniGraph2 is a unified framework for multimodal graph representation learning, supporting both node classification and link prediction tasks. The model leverages pre-trained encoders (T5-ViT and CLIP) for feature extraction and employs a Mixture of Experts (MoE) architecture for effective multimodal alignment.

## Features

- **Multimodal Support**: Handles both text and image modalities using pre-trained encoders
- **Flexible Architecture**: Mixture of Experts (MoE) for dynamic feature alignment
- **Multiple Tasks**: Supports both node classification and link prediction
- **Efficient Training**: PyTorch Lightning integration for scalable training
- **Experiment Tracking**: Weights & Biases integration for experiment monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/unigraph2.git
cd unigraph2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
unigraph2/
├── data/
│   ├── datamodule.py      # Data loading and preprocessing
│   ├── nc_dataset.py      # Node classification dataset
│   └── lp_dataset.py      # Link prediction dataset
├── models/
│   └── unigraph2.py       # UniGraph2 model implementation
├── train.py               # Training script
└── requirements.txt       # Project dependencies
```

## Usage

### Data Preparation

1. Prepare your graph data in the following format:
   - `graph.bin`: DGL graph structure
   - `features_t5vit.pt`: Text features (T5-ViT)
   - `features_clip.pt`: Image features (CLIP)
   - `train_mask.pt`, `val_mask.pt`, `test_mask.pt`: Node masks for splits
   - `labels.pt`: Node labels (for node classification)

2. Place your data in the `data/example` directory.

### Training

To train the model, simply run:

```bash
python train.py
```

The training script will:
- Load and preprocess the data
- Initialize the UniGraph2 model
- Train with automatic mixed precision
- Save checkpoints and log metrics to Weights & Biases

### Configuration

The model can be configured by modifying the parameters in `train.py`:

```python
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
)
```


## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{he2025unigraph2,
  title={UniGraph2: Learning a Unified Embedding Space to Bind Multimodal Graphs},
  author={He, Yufei and Sui, Yuan and He, Xiaoxin and Liu, Yue and Sun, Yifei and Hooi, Bryan},
  booktitle={THE WEB CONFERENCE 2025}
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


