# Instructions to Download and Setup the NFL Dataset from Kaggle
The NFL Big Data Bowl 2025 dataset is protected under a competition-specific license. To ensure proper usage and compliance with Kaggle competitions' data sharing policies, this process is necessary.
The data can only be accessed through Kaggle's API after accepting the competition rules and terms.


## Prerequisites
1. Kaggle account (create one at [kaggle.com](https://kaggle.com) if you don't have one)
2. Kaggle API credentials

## Setup Steps

### 1. Install the Kaggle API
```bash
pip install kaggle
```

### 2. Authenticate with the Kaggle API
1. Go to your Kaggle account settings (https://www.kaggle.com/settings)
2. Scroll to "API" section and click "Create New API Token"
3. This will download a `kaggle.json` file
4. Place the `kaggle.json` file in one of these locations:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`
5. Set appropriate permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 3. Download the Dataset
```bash
kaggle competitions download -c nfl-big-data-bowl-2025
unzip nfl-big-data-bowl-2025.zip -d nfl-big-data-bowl-2025
```

### 4. Configure Project Settings
Update your project configuration (found in `etl/config.py`) to match your local setup.

### 5. Verify Dataset Structure
After setup, ensure you have the following CSV files in your dataset directory:
- `games.csv`
- `players.csv`
- `player_play.csv`
- `plays.csv`
- `tracking_week_{1-9}.csv` (tracking data files for each week)

