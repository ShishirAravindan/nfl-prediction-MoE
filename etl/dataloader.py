"""
Dataloaders for expert models and combined dataset.

This module provides dataset classes for:
1. Individual expert models (CNN, MLP, RNN) via ExpertDataset
2. Combined features across all experts via CombinedExpertDataset

Key functionality:
- Loads features from respective sources (npy/parquet files)
- Handles key-based filtering (gameId, playId)
- Ensures data consistency across experts
- Converts features to appropriate tensor formats
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# given a dataframe of (gameId, playId, target) filter the features for an expert (where (gameId, playId) is the key) to create a dataset.

class ExpertDataset(Dataset):
    def __init__(self, df, expert_name):
        self.expert_name = expert_name
        self.keys = df[['gameId', 'playId']].drop_duplicates().values
        self._key_set = set(map(tuple, self.keys))
        self.targets = df['isPass'].values
        
        self.features = self.load_features() 
        # NOTE: remove the key, target columns where data is missing for feature to ensure consistency

    def load_features(self):
        if self.expert_name == "cnn":
            return self.load_cnn_features()
        elif self.expert_name == "mlp":
            return self.load_mlp_features()
        elif self.expert_name == "rnn":
            return self.load_rnn_features()
        
    def load_cnn_features(self):
        cnn_features = np.load('../demo/features/cnn_features.npy', allow_pickle=True)
        filtered_features = []
        valid_indices = []
        # For each key in our key_set
        for idx, key in enumerate(self._key_set):
            # Find the matching row in cnn_features where gameId and playId match
            mask = (cnn_features[:, 0] == key[0]) & (cnn_features[:, 1] == key[1])
            matching_row = cnn_features[mask]
            
            if len(matching_row) > 0:
                # Append the feature (assuming it's in position 2)
                filtered_features.append(matching_row[0, 2])
                valid_indices.append(idx)

        # Update the keys and targets to only include those with valid features
        self.keys = self.keys[valid_indices]
        self.targets = self.targets[valid_indices]
                
        return np.array(filtered_features)
    
    def load_mlp_features(self):
        mlp_features = pd.read_parquet('../demo/features/mlp_features.parquet')
        filtered_features = []
        valid_indices = []
        for idx, key in enumerate(self._key_set):
            mask = (mlp_features['gameId'] == key[0]) & (mlp_features['playId'] == key[1])
            matching_row = mlp_features[mask]
            if len(matching_row) > 0:
                # get the features
                feature_columns = ["week", "down", "quarter", "yardsToGo", 
                                   "yardlineNumber", "possessionTeam", "receiverAlignment", 
                                   "preSnapHomeScore", 'defensiveTeam', 'yardlineSide',
                                   "preSnapVisitorScore", "preSnapHomeTeamWinProbability", "pff_runPassOption",
                                   "preSnapVisitorTeamWinProbability", "playClockAtSnap"]
                # Append the features
                filtered_features.append(matching_row[feature_columns].values[0])
                valid_indices.append(idx)
        # Update the keys and targets to only include those with valid features
        self.keys = self.keys[valid_indices]
        self.targets = self.targets[valid_indices]
        
        return np.array(filtered_features)

    def load_rnn_features(self):
        rnn_features = pd.read_parquet('../demo/features/rnn_features.parquet')
        filtered_features = []
        valid_indices = []
        # For each key in our key_set
        for idx, key in enumerate(self._key_set):
            # Find the matching row in rnn_features where gameId and playId match
            mask = (rnn_features['gameId'] == key[0]) & (rnn_features['playId'] == key[1])
            matching_row = rnn_features[mask]
            
            if len(matching_row) > 0:
                # get the features
                feature_names = ["relative_x", "relative_y", "s", "a", "dis", "o", "dir"]
                sequence_features = []
                for feature_name in feature_names:
                    feature_data = matching_row[feature_name].values[0]
                    # Convert from numpy.object_ to numpy.float32
                    feature_data = np.array([np.array(player_data, dtype=np.float32) 
                                           for player_data in feature_data], dtype=np.float32)
                    sequence_features.append(feature_data)
                    
                # Stack all features into a single array
                sequence_features = np.stack(sequence_features, axis=-1)
                filtered_features.append(sequence_features)
                valid_indices.append(idx)
                
        # Update the keys and targets to only include those with valid features
        self.keys = self.keys[valid_indices]
        self.targets = self.targets[valid_indices]

        return np.array(filtered_features)

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        """
        Returns a single sample and its target at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (feature, target) where feature is the input data 
                  and target is the corresponding label
        """
        feature = self.features[idx]
        target = self.targets[idx]
        
        # Convert to torch tensors if they aren't already
        feature = torch.FloatTensor(feature)
        target = torch.FloatTensor([target])
        
        return feature, target

class CombinedExpertDataset(Dataset):
    """Dataset combining features from all experts.
    
    Only includes samples that have valid features across all experts.
    
    Attributes:
        keys (list): List of (gameId, playId) tuples valid for all experts
        cnn_features (torch.Tensor): Features for CNN expert
        mlp_features (torch.Tensor): Features for MLP expert
        rnn_features (torch.Tensor): Features for RNN expert
        targets (torch.Tensor): Binary target values
    """

    def __init__(self, df):
        """
        Dataset that combines inputs for all experts while preserving keys
        Args:
            df: DataFrame containing all features
        """
        # Create individual expert datasets
        self.cnn_dataset = ExpertDataset(df, expert_name="cnn")
        self.mlp_dataset = ExpertDataset(df, expert_name="mlp")
        self.rnn_dataset = ExpertDataset(df, expert_name="rnn")
        
        # Get intersection of valid keys across all experts
        cnn_keys = set([tuple(key) for key in self.cnn_dataset.keys])
        mlp_keys = set([tuple(key) for key in self.mlp_dataset.keys])
        rnn_keys = set([tuple(key) for key in self.rnn_dataset.keys])
        
        # Only keep keys that exist in all experts
        self.valid_keys = sorted(list(cnn_keys.intersection(mlp_keys, rnn_keys)))
        
        # Filter features and targets for each expert to only include valid keys
        self.cnn_features = []
        self.mlp_features = []
        self.rnn_features = []
        self.targets = []
        
        # Create filtered datasets using only valid keys
        for key in self.valid_keys:
            cnn_idx = np.where((self.cnn_dataset.keys == np.array(key)).all(axis=1))[0][0]
            mlp_idx = np.where((self.mlp_dataset.keys == np.array(key)).all(axis=1))[0][0]
            rnn_idx = np.where((self.rnn_dataset.keys == np.array(key)).all(axis=1))[0][0]
            
            self.cnn_features.append(self.cnn_dataset.features[cnn_idx])
            self.mlp_features.append(self.mlp_dataset.features[mlp_idx])
            self.rnn_features.append(self.rnn_dataset.features[rnn_idx])
            self.targets.append(self.cnn_dataset.targets[cnn_idx])
        
        # Convert to tensors if not already
        self.cnn_features = torch.tensor(self.cnn_features, dtype=torch.float32) / 255.0  # normalize to [0,1]
        self.mlp_features = torch.tensor(self.mlp_features, dtype=torch.float32)
        self.rnn_features = torch.tensor(self.rnn_features, dtype=torch.float32)
        self.targets = torch.FloatTensor(self.targets).unsqueeze(1)
        
    def __len__(self):
        return len(self.valid_keys)
    
    def __getitem__(self, idx):
        key = self.valid_keys[idx]
        
        batch_data = {
            'cnn': self.cnn_features[idx],
            'mlp': self.mlp_features[idx],
            'rnn': self.rnn_features[idx]
        }
        
        return key, batch_data, self.targets[idx]