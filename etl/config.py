PROJECT_DIR = '/content/drive/.shortcut-targets-by-id/1vcMqu6hVQWEi3qkOsgMGufzBcFhrLLBN/CSC413'
DATASET_DIR = f'{PROJECT_DIR}/nfl-big-data-bowl-2025/'
TARGET_FILE_PATH = DATASET_DIR + "features/target.csv"
FILES = {
    "games": DATASET_DIR + "games.csv",
    "players": DATASET_DIR + "players.csv",
    "player_play": DATASET_DIR + "player_play.csv",
    "plays": DATASET_DIR + "plays.csv",
    "week": lambda week: DATASET_DIR + f"tracking_week_{week}.csv"
}

CLUB_COLORS = {
    'MIN': "#FF0000",  # Bright Red
    'PHI': "#FFFF00",  # Bright Yellow
    'football': "#FFFFFF",  # White
    'BUF': "#0000FF",  # Bright Blue
    'TEN': "#FF7F00",  # Bright Orange
    'GB': "#FF00FF",  # Magenta
    'CHI': "#00FFFF",  # Cyan
    'ARI': "#800000",  # Maroon
    'LV': "#808000",  # Olive
    'DEN': "#008080",  # Teal
    'HOU': "#800080",  # Purple
    'DAL': "#FFA500",  # Deep Orange
    'CIN': "#5F9EA0",  # Cadet Blue
    'SF': "#FFD700",  # Gold
    'SEA': "#00FF00",  # Bright Green
    'LA': "#4682B4",  # Steel Blue
    'ATL': "#FF1493",  # Deep Pink
    'NE': "#7FFF00",  # Chartreuse
    'PIT': "#DC143C",  # Crimson
    'CAR': "#ADFF2F",  # Green Yellow
    'NYG': "#6495ED",  # Cornflower Blue
    'NO': "#DDA0DD",  # Plum
    'TB': "#FF6347",  # Tomato
    'IND': "#9ACD32",  # Yellow Green
    'JAX': "#FF4500",  # Orange Red
    'WAS': "#2E8B57",  # Sea Green
    'DET': "#40E0D0",  # Turquoise
    'NYJ': "#B22222",  # Firebrick
    'CLE': "#6A5ACD",  # Slate Blue
    'BAL': "#00BFFF",  # Deep Sky Blue
    'MIA': "#F4A460",  # Sandy Brown
    'KC': "#8A2BE2",  # Blue Violet
    'LAC': "#FFC0CB",  # Pink
}
