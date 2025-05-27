from PIL import Image
import numpy as np
#from test_network import create_model

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def show(src, reseau, X, Y):
    size = 2
    zoom = 1
    startX = -2 * size
    startY = -2 * size
    width = 4 * size
    height = 4 * size
    multi = 20 * zoom
    increment = 0.050 / zoom
    threshold = 0.5

    img = Image.new('RGB', (width * multi, height * multi), 'white')
    pixels = img.load()

    for j in np.arange(startX, startX + width + increment, increment):
        # Get predicted y value
        pred_y, Z_val = reseau.forward_propagation(np.array([j]).reshape(-1, 1))
        pred_y = pred_y[-1]
        pred_y = pred_y.item()

        # Convert coordinates to pixel space
        x = int((j - startX) * multi)
        y = img.height - int((pred_y - startY) * multi) - 1
        
        if 0 <= x < img.width and 0 <= y < img.height:
            pixels[x, y] = (255, 0, 0)  # Red line for prediction

    X, Y = X.T, Y.T

    for i in range(len(X)):
            data_x, data_y = X[i], Y[i]
            # Convert to integers and flip y-coordinate
            px = int((data_x - startX) * multi)
            py = img.height - int((data_y - startY) * multi) - 1
            if 0 <= px < img.width and 0 <= py < img.height:
                pixels[px, py] = (225, 105, 65)

    img.save(src)
