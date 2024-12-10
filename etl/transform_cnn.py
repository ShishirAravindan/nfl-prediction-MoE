import pandas as pd
import pickle
import numpy as np
from nfl import visuals
import matplotlib.pyplot as plt
from config import *

def generate_formation_image(data, gameId, playId, frameId):
    """Given a single play, generate a formation image."""
    fig, ax = visuals.snap(data, gameId, playId, frameId,
                           fifty_yard=True, size=250, club_colors=CLUB_COLORS)

    fig.canvas.draw() # Render the figure
    image = np.array(fig.canvas.renderer.buffer_rgba())

    plt.close(fig) # Close the figure

    return image

def get_formation_datapoints(weekNum, TARGETS):
    """Given a week number and its corresponding targets, 
    generate a list of datapoints for formation images."""
    week = pd.read_csv(FILES["week"](weekNum))
    vizData = []

    unique_frames = week[week['frameType'] == 'SNAP']
    unique_frames = unique_frames[['gameId', 'playId', 'frameId']].drop_duplicates().to_numpy()
    print(f'There are {len(unique_frames)} frames to generate visualizations for week {weekNum}')

    i = 0

    for row in unique_frames:
        i += 1
        print(f'Row {i} processed')
        gameId, playId, frameId = row[0], row[1], row[2]
        target = TARGETS.get((gameId, playId), None)
        image = generate_formation_image(week, gameId, playId, frameId)

        vizData.append((gameId, playId, image, target))

    vizData = np.array(vizData, dtype=object)
    return vizData

def transform_all_formation_images():
    pass