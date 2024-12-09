import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from config import *

def get_offensive_plays(df_week):
    """Filter such that only offensive frames are considered"""
    df_plays = pd.read_csv(FILES["plays"])
    df_week = df_week.merge(df_plays[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"], how="left")
    df_week = df_week[df_week["possessionTeam"] == df_week["club"]]
    df_week = df_week.drop(columns=["possessionTeam"])
    return df_week

def create_timeToSnap_column(df_week):
    """
    Filters the weekly tracking data where the frame type is snap. 
    Merges back snap time for each play
    Computes timeToSnap and drops other unused time-related columns."""
    df_week['time'] = pd.to_datetime(df_week['time'], format='mixed', errors='coerce')

    snaps = df_week[df_week['frameType'] == 'SNAP']
    snaps = snaps[['playId', 'time']].rename(columns={'time': 'snap_time'})
    # duplicates_before = snaps.duplicated(subset=['playId']).sum()
    snaps = snaps.drop_duplicates(subset=['playId'])
    # ---
    print(f"{snaps['snap_time'].isna().sum()} timestamps with error.")
    # print(f"Duplicates found and dropped: {duplicates_before}")
    # ---

    # Consider only pre-snap frames
    df_week = df_week[df_week['frameType'] == 'BEFORE_SNAP']
    df_week = df_week.merge(snaps, on='playId', how='left')
    df_week['timeToSnap'] = (df_week['snap_time'] - df_week['time']).dt.total_seconds()

    # drop unused columns
    df_week = df_week.drop(columns=['snap_time', 'time', 'frameId',
                                    'playDirection', 'jerseyNumber', 'frameType'])

    return df_week


def create_role_column(df):
    # add a new column named role that tells us what poistion the players play
    # use a embedding layers
    POSITIONS = ['QB', 'T', 'TE', 'WR', 'DE', 'NT', 'SS', 'FS', 'G', 'OLB', 'DT', 'CB', 'RB', 'C', 'ILB', 'MLB', 'FB', 'DB', 'LB']
    DEFENSIVE_POSITIONS = ['DE', 'NT', 'SS', 'FS', 'OLB', 'DT', 'CB', 'ILB', 'MLB', 'DB', 'LB']

    OFFENSIVE_POSITIONS = ['QB', 'T', 'TE', 'WR', 'G', 'RB', 'C', 'FB']

    df_players = pd.read_csv(FILES["players"])
    df = df.merge(df_players[["nflId", "position"]], on="nflId", how="left")

    # filter offensive players
    df_offensive = df[df['position'].isin(OFFENSIVE_POSITIONS)]

    # Map positions to unique IDs
    position_to_id = {pos: idx for idx, pos in enumerate(OFFENSIVE_POSITIONS)}
    df_offensive['position_id'] = df_offensive['position'].map(position_to_id)

    # Create embedding layer
    num_positions = len(OFFENSIVE_POSITIONS)  # Number of unique offensive positions
    embedding_dim = 8  # Dimensionality of embedding(hyper parameter)
    embedding_layer = nn.Embedding(num_positions, embedding_dim)

    # Convert position IDs to tensors
    position_ids = torch.tensor(df_offensive['position_id'].values, dtype=torch.long)
    embedded_positions = embedding_layer(position_ids) # hyper pararmeter

    # drop players name and frameId
    df_offensive = df_offensive.drop(["displayName"], axis=1)

    return df

def create_event_one_hot(df_week):
    """Generate one-hot encoding of pre-snap events"""
    df_week.loc[:, 'eventFilled'] = df_week.groupby('playId')['event'].ffill()

    # Creating mapping of event to embedding
    # unique_events = df_week_1['eventFilled'].unique()
    UNIQUE_EVENTS = ['huddle_break_offense', 'line_set', 'man_in_motion',
                     'shift', 'timeout_away', 'no_event']

    # Generates the one-hot encoded list for each row
    def _get_event_one_hot(event, UNIQUE_EVENTS):
        if pd.isna(event): event = 'no_event'
        return [1 if event == unique_event else 0 for unique_event in UNIQUE_EVENTS]

    # Apply the function to create the one-hot encoded lists and add it as a new column
    df_week.loc[:, 'event_vector'] = df_week['eventFilled'].apply(
        lambda x: _get_event_one_hot(x, UNIQUE_EVENTS))

    df_week = df_week.drop(columns=['event', 'eventFilled'])

    return df_week

def remove_small_plays(df):
    # group by (gameId, playId, nflId) to get the frames for a player i.e. frameId
    df = df.groupby(['gameId', 'playId', 'nflId']).filter(lambda x: len(x) >= 10)
    return df

def get_ten_frames(df):
    def sample_interval(group):
        num_frames = len(group)
        if num_frames <= 10:
            return group  # Expect this never to happen

        interval = num_frames // 10
        indices = [i * interval for i in range(10)]

        return group.iloc[indices].reset_index(drop=True)  # Return the sampled frames

    df = df.groupby(['gameId', 'playId', 'nflId', 'position']).apply(sample_interval).reset_index(drop=True)
    # remove gameId, playId, nflId
    # df = df.drop(columns=['gameId', 'playId', 'nflId', 'position'])
    return df
