# ETL Module Documentation

This module handles the Extract, Transform, Load (ETL) pipeline for the NFL Big Data Bowl 2025 dataset, focusing on pass/rush prediction given pre-snap play data.

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
```

## Module Structure

### Data Extraction (`extract.md`)
- Instructions for downloading the NFL dataset from Kaggle

### Configuration (`config.py`)
- Defines global constants and paths
- Contains dataset file locations and team color mappings
- Configures project-wide settings


### Feature Engineering (`feature-engineering-utils.py`)
Core utilities for data preprocessing:
- `get_offensive_plays()`: Filters for offensive frames
- `create_timeToSnap_column()`: Computes time-to-snap metrics
- `create_role_column()`: Adds position embeddings
- `create_event_one_hot()`: One-hot encodes pre-snap events
- `remove_small_plays()`: Filters out plays with insufficient frames
- `get_ten_frames()`: Standardizes frame count per play

### Transformation Pipelines

#### CNN Pipeline (`transform_cnn.py`)
Handles formation image generation:
- Converts play data into visual representations
- Generates standardized formation images for CNN input

#### MLP Pipeline (`transform_mlp.py`)
Processes tabular features for MLP model:
- Performs feature selection and handles categorical encoding

#### RNN Pipeline (`transform_rnn.py`)
Processes sequential data for RNN model:
- Implements feature engineering pipeline
- Converts play sequences into tensor format
- Handles multi-feature temporal data

## Usage

1. First, follow the setup instructions in `extract.md` to download the dataset
2. Configure paths in `config.py`
3. Use the appropriate transformation pipeline based on your expert-model

## Data Flow

```
Raw Data → transformation pipeline → Model-Specific Training Data
```