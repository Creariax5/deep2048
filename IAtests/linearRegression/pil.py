from PIL import Image
import numpy as np
from main import get_dataset
import pandas as pd # data processing

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def show(src, reseau):
    startX = 0
    startY = -1
    width = 8
    height = 4
    multi = 20
    increment = 0.05

    img = Image.new('RGB', (width * multi, height * multi), 'white')
    pixels = img.load()

    for j in np.arange(startX, startX + width + increment, increment):
        # Get predicted y value
        pred_y = reseau.forward_propagation([j])
        # Convert coordinates to pixel space
        x = int((j - startX) * multi)
        y = img.height - int((pred_y - startY) * multi) - 1
        
        if 0 <= x < img.width and 0 <= y < img.height:
            pixels[x, y] = (255, 0, 0)  # Red line for prediction

    # Plot data points
    data_set = get_dataset()
    for i in range(len(data_set)):
        row = data_set.iloc[i]
        data_x, data_y = row.values[:3]
        px = int((data_x - startX) * multi)
        py = img.height - int((data_y - startY) * multi) - 1
        if 0 <= px < img.width and 0 <= py < img.height:
            pixels[px, py] = (0, 0, 255)  # Blue dots for data points

    img.save(src)