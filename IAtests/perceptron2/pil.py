from PIL import Image
import numpy as np
<<<<<<< HEAD
from main import learning_speed_test, get_dataset, create_single_neuron
import pandas as pd # data processing
import numpy as np
=======
from main import learning_speed_test, get_dataset
import pandas as pd # data processing
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from perceptronSimple import PerceptronSimple

<<<<<<< HEAD
threshold = 0.5

def show(src, reseau):
    data_offset = 0
    startX = 1 + data_offset
    startY = 0 + data_offset
    width = 7
    height = 3
    multi = 20
    increment = 0.05
=======
def show(src, reseau):
    startX = 0
    startY = -1
    width = 8
    height = 4
    multi = 20
    increment = 0.05
    threshold = 0.5
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5

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
<<<<<<< HEAD
                res = reseau.forward_propagation([i, j])
                pixels[x, y] = (65, 105, int(225*res)) if res < threshold else (int(255*res), 215, 0)
=======
                res = reseau.forward_propagation([j, i])
                pixels[x, y] = (int(255*res), int(225*res), 225) if res < threshold else (int(255*res), int(225*res), 0)
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5

    data_set = get_dataset()
    for i in range(len(data_set)):
            row = data_set.iloc[i]
            data_x, data_y, type = row.values[:3]
            # Convert to integers and flip y-coordinate
<<<<<<< HEAD
            px = int((data_x - startX + data_offset) * multi)
            py = img.height - int((data_y - startY + data_offset) * multi) - 1
=======
            px = int((data_x - startX) * multi)
            py = img.height - int((data_y - startY) * multi) - 1
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5
            if 0 <= px < img.width and 0 <= py < img.height:
                pixels[px, py] = (225, 105, 65) if type == 0 else (0, 215, 255)

    img.save(src)


<<<<<<< HEAD
neurone = create_single_neuron(2)
print("Initial weights:", neurone.w)
=======
neurone = PerceptronSimple(2)
print("Initial weights:", neurone.w, neurone.b)
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5
    
show('not_trained.png', neurone)
    
print("Training perceptron...")
<<<<<<< HEAD
neurone = learning_speed_test(neurone, 0.01, 150)
print("Final weights:", neurone.w)
=======
neurone = learning_speed_test(neurone, 5, 150)
print("Final weights:", neurone.w, neurone.b)
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5
    
show('trained.png', neurone)
