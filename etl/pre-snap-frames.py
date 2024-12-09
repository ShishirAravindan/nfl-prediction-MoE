from . import feature_engineering_utils as fe_utils

import pandas as pd
import numpy as np
import torch

from config import *


def feature_engineering(df_week):
    # filter for only defensive frames
    df_week = fe_utils.get_offensive_plays(df_week)

    # Engineer features
    df_week = fe_utils.create_timeToSnap_column(df_week) # filters to pre-snap frames plays only + removes unused columns
    df_week = fe_utils.create_role_column(df_week)
    df_week = fe_utils.create_event_one_hot(df_week)

    # Remove small plays and standardize number of frames
    df_week = fe_utils.remove_small_plays(df_week)
    df_week = fe_utils.get_ten_frames(df_week)

    return df_week


def get_datapoints(df_week, TARGET_FILE_PATH):
    df_week = feature_engineering(df_week)


    X = df_week.drop(columns=["displayName", "club", "position"]) # drop "displayName", "club", "position"
    # create an index of ('gameId', 'playId', 'nflId', 'position')
    Y = X.groupby(['gameId', 'playId', 'nflId']).agg(list).reset_index()
    # join the table with play on nflId to get the position of each plauers
    P = pd.read_csv(FILES["players"])
    Y = Y.merge(P[["nflId", "position"]], on="nflId", how="left")

    Z = Y.groupby(['gameId', 'playId']).agg(list)
    T = pd.read_csv(TARGET_FILE_PATH)
    df_plays = pd.read_csv(FILES["plays"])
    F = Z.merge(df_plays[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"], how="left")
    F = F.merge(T, on=["gameId", "playId"], how="left")

    return F

def get_feature_tensor(feature_data):
    """
    Converts feature data from numpy.object_ type to torch.tensor with dtype=float32,
    then transposes and reshapes it to (timesteps, players * features).
    """
    # Convert from numpy.ndarray type: from numpy.object_ to numpy.float32
    feature_data = np.array([np.array(player_data, dtype=np.float32) for player_data in feature_data], dtype=np.float32)

    # Convert into torch.tensor with dtype=float32
    feature_tensor = torch.tensor(feature_data, dtype=torch.float32)

    # Transpose to (timesteps, players, features)
    feature_tensor = feature_tensor.transpose(0, 1)

    # Reshape to (timesteps, players * features) by flattening the second dimension
    feature_tensor = feature_tensor.reshape(feature_tensor.shape[0], -1)  # Shape: (timesteps, players * features)

    return feature_tensor

def load_and_process_data(df):
    inputs = []
    targets = []

    for _, row in df.iterrows():
        # STEP 1: Convert input data into feature tensors --> numpy.object_ to torch.tensor with dtpe=float32
        # Creates a feature tensor of shape (timestep, num_features) i.e. (10, 11)
        feature_names = ["x", "y", "s", "a", "dis", "o", "dir", "timeToSnap"]
        processed_features = []
        for feature_name in feature_names:
            feature_tensor = get_feature_tensor(row[feature_name])
            processed_features.append(feature_tensor)

        # STEP 2: Aggregate for all (8) features of a play. Stack all features
        # along the second axis (players * features).
        # Shape: (timesteps, players * features) i.e. (10, 11 * 8)
        input_tensor = torch.cat(processed_features, dim=1)

        # Assuming the target is in the "isPass" column (one per play)
        target = torch.tensor(row["isPass"], dtype=torch.float32)  # dtype=torch.long for integer class labels

        # STEP 3: Append the processed play and target to the list
        inputs.append(input_tensor)
        targets.append(target)

    # Convert inputs and targets to tensors
    inputs = torch.stack(inputs)  # Shape: (plays, timesteps, players * features)
    targets = torch.stack(targets)  # Shape: (plays,)

    return inputs, targets