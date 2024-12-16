import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from etl.config import *

def convert_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds


def get_features_splits(game_df, play_df, target):    
    # 1. Select relevant pre-snap columns
    pre_snap_cols = ["gameId", "playId", "week", "down", "quarter", "yardsToGo", "yardlineNumber",
                 "possessionTeam", "receiverAlignment", "preSnapHomeScore", 'defensiveTeam', 'yardlineSide',
                 "preSnapVisitorScore", "preSnapHomeTeamWinProbability", "pff_runPassOption",
                 "preSnapVisitorTeamWinProbability", "playClockAtSnap", 'target']

    # 2. Merge datasets
    # Ensure merging on gameId and playId for Play data
    df = play_df.merge(game_df, on="gameId", how="inner")
    #left join on target using playid or gameid
    df = df.merge(target, on=["gameId", "playId"], how="left")
    #df = df.merge(player_play_df, on=["gameId", "playId"], how="inner")

    df = df[pre_snap_cols]
    # 3. get rid of nan
    df = df.dropna()
    # 4. Encode categorical variables
    categorical_cols = ["possessionTeam", "defensiveTeam", "yardlineSide", "receiverAlignment"]
    label_encoders = {}

    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])


    X_df = df.drop(columns=["target"])
    y_pass = df["target"]
    #make y_pass one hot
    y_pass = pd.get_dummies(y_pass)

    # 6. Ensure all features are numerical
    assert X_df.select_dtypes(include=["object"]).empty, "Not all features are numerical!"


    X_valid = X_df[int(2*X_df.shape[0]/3):].values
    y_valid = y_pass[int(2*X_df.shape[0]/3):].values


    X_train, X_test, y_train, y_test = train_test_split(X_df[:int(2*X_df.shape[0]/3)], y_pass[:int(2*X_df.shape[0]/3)], test_size=0.2, random_state=42)
    # 7. Split data into training and testing sets
    # transform into numpy arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def load_mlp_features():
    game_df = pd.read_csv(FILES["games"])
    play_df = pd.read_csv(FILES["plays"])
    target = pd.read_csv(FILES["target"]) 
    return get_features_splits(game_df, play_df, target)