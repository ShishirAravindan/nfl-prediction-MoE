"""
- create a dataloader for each expert. 
- Look at target.csv to generate a test train split based on (gameId, playId).
- Use (gameId, playId) as key to find the corresponding features for each expert.
    - CNN: see demo/features/cnn_features.npy which is a numpy array of shape (gameId, playId, CNN_features, target)
        - CNN_features: an image
    - MLP: see demo/features/mlp_features.parquet which is a pandas dataframe with columns (gameId, playId, MLP_features, target)
        - MLP_features: a dataframe of 15 features
    - RNN: see demo/features/rnn_features.parquet which is a pandas dataframe with columns (gameId, playId, RNN_features, target)
        - RNN_features: a dataframe of _ features
- the dataloader should take the expert name as argument to know which features to load
- the get function should take the gameId, playId as argument to get the features
- ideally three dataloaders should be created, one for each expert where the gameId, playId is the key.
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
        self.key_set = set(map(tuple, self.keys))
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
        for idx, key in enumerate(self.key_set):
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
        for idx, key in enumerate(self.key_set):
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
        for idx, key in enumerate(self.key_set):
            # Find the matching row in rnn_features where gameId and playId match
            mask = (rnn_features['gameId'] == key[0]) & (rnn_features['playId'] == key[1])
            matching_row = rnn_features[mask]
            
            if len(matching_row) > 0:
                # Get the sequence features (x, y, s, a, dis, o, dir, timeToSnap)
                feature_names = ["x", "y", "s", "a", "dis", "o", "dir", "timeToSnap"]
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
