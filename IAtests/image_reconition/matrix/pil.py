from PIL import Image
import numpy as np
#from test_network import create_model

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def show(src, reseau, X, Y):
    size = 3
    zoom = 1
    startX = -1 * size
    startY = -1 * size
    width = 2 * size
    height = 2 * size
    multi = 20 * zoom
    increment = 0.050 / zoom
    threshold = 0.5

    img = Image.new('RGB', (width * multi, height * multi), 'white')
    pixels = img.load()

    for i in np.arange(startY, startY + height + increment, increment):
        for j in np.arange(startX, startX + width + increment, increment):
            # Convert to integers for pixel coordinates
            x = int((j - startX) * multi)
            # Flip the y-coordinate by subtracting from the height
            y = img.height - int((i - startY) * multi) - 1
            # Check bounds before setting pixel
            if 0 <= x < img.width and 0 <= y < img.height:
                l = [j, i]
                res = reseau.forward_propagation(np.array(l).reshape(-1, 1))
                # print(len(res))
                res = res[len(res) - 1]
                # print(res.shape)
                # Extract the scalar value from the numpy array
                res_value = res.item()  # or res[0,0] depending on your array shape
                pixels[x, y] = (int(255*res_value), int(225*res_value), 225) if res_value < threshold else (int(255*res_value), int(225*res_value), 0)

    X, Y = X.T, Y.T

    for i in range(len(X)):
            data_x, data_y = X[i]
            type = Y[i]
            # Convert to integers and flip y-coordinate
            px = int((data_x - startX) * multi)
            py = img.height - int((data_y - startY) * multi) - 1
            if 0 <= px < img.width and 0 <= py < img.height:
                pixels[px, py] = (225, 105, 65) if type == 0 else (0, 215, 255)

    img.save(src)
