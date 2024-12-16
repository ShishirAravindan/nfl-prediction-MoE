# ETL Module Documentation

This module handles the Extract, Transform, Load (ETL) pipeline for the NFL Big Data Bowl 2025 dataset, focusing on pass/rush prediction given pre-snap play data.

## Quick Start with Pre-generated Features

For quick experimentation, you can download pre-generated features from our Google Drive:
1. Download features from [Google Drive](https://drive.google.com/drive/folders/13pSIpgwfowXHJcbAjYNnEJwuUUyNFTb-?usp=drive_link)
2. Place the downloaded files in the `demo/features/` directory:
   ```
   demo/features/
   ├── cnn_features.npy
   ├── mlp_features.parquet
   └── rnn_features.parquet
   ```
3. Use `dataloader.py` to load the features into a PyTorch dataset to train models.

Alternatively, you can use the model checkpoints from the `checkpoints/` directory which contain pre-trained weights.

---
## Full Dataset Processing

If you want to process the raw NFL data yourself, follow the instructions below:

## File Structure
```
etl/
├── README.md
├── config.py
├── extract.md
├── feature-engineering-utils.py
├── transform_cnn.py
├── transform_mlp.py
└── transform_rnn.py
└── dataloader.py
```

## Module Structure

### Extract
**Data Extraction (`extract.md`)**
- Instructions for downloading the NFL dataset from Kaggle

**Configuration (`config.py`)**
- Defines global constants and paths
- Contains dataset file locations and team color mappings
- Configures project-wide settings

### Transform
**Feature Engineering (`feature-engineering-utils.py`)**
Core utilities for data preprocessing:
- `get_offensive_plays()`: Filters for offensive frames
- `create_timeToSnap_column()`: Computes time-to-snap metrics
- `create_role_column()`: Adds position embeddings
- `create_event_one_hot()`: One-hot encodes pre-snap events
- `remove_small_plays()`: Filters out plays with insufficient frames
- `get_ten_frames()`: Standardizes frame count per play

**CNN Pipeline (`transform_cnn.py`)**
Handles formation image generation:
- Converts play data into visual representations
- Generates standardized formation images for CNN input

**MLP Pipeline (`transform_mlp.py`)**
Processes tabular features for MLP model:
- Performs feature selection and handles categorical encoding

**RNN Pipeline (`transform_rnn.py`)**
Processes sequential data for RNN model:
- Implements feature engineering pipeline
- Converts play sequences into tensor format
- Handles multi-feature temporal data

### Load
**Data Loader (`dataloader.py`)**
- Creates datasets for each expert model

## Usage

1. First, follow the setup instructions in `extract.md` to download the dataset
2. Configure paths in `config.py`
3. Use the appropriate transformation pipeline based on your expert-model

## Data Flow

```
Raw Data → transformation pipeline → Model-Specific Training Dataset
```

## Next Steps
- See demos on how experts are trained (and loaded if pretrained)
- See how MoE is trained