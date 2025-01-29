from PIL import Image
import numpy as np
from main import learning_speed_test, get_dataset
import pandas as pd # data processing
import numpy as np

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from IAtests.perceptron.perceptron1957 import Perceptron1957

def show(src, reseau):
    data_offset = -4
    startX = 1 + data_offset
    startY = 0 + data_offset
    width = 7
    height = 3
    multi = 20
    increment = 0.05

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
                pixels[x, y] = (65, 105, 225) if reseau.forward_propagation(i, j) == 0 else (255, 215, 0)

    data_set = get_dataset()
    for i in range(len(data_set)):
            row = data_set.iloc[i]
            data_x, data_y, type = row.values[:3]
            # Convert to integers and flip y-coordinate
            px = int((data_x - startX + data_offset) * multi)
            py = img.height - int((data_y - startY + data_offset) * multi) - 1
            if 0 <= px < img.width and 0 <= py < img.height:
                pixels[px, py] = (225, 105, 65) if type == 0 else (0, 215, 255)

    img.save(src)


neurone = Perceptron1957(w1=np.random.uniform(-1, 1), w2=np.random.uniform(-1, 1))
print("Initial weights:", neurone.w1, neurone.w2)
    
show('not_trained.png', neurone)
    
print("Training perceptron...")
neurone = learning_speed_test(neurone, 0.01, 150)
print("Final weights:", neurone.w1, neurone.w2)
    
show('trained.png', neurone)
