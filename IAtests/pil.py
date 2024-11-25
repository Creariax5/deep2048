from PIL import Image
from perceptron.perceptron1957 import Perceptron1957
import random

nb = 3

neurone1 = Perceptron1957(random.randint(-nb, nb), random.randint(-nb, nb))
neurone2 = Perceptron1957(random.randint(-nb, nb), random.randint(-nb, nb))
neurone3 = Perceptron1957(random.randint(-nb, nb), random.randint(-nb, nb))

def get_res(x, y):
    res1 = neurone1.run(x, y)
    res2 = neurone2.run(x, y)
    return neurone3.run(res1, res2)

size = 300

img = Image.new('RGB', (size * 2 + 1, size * 2 + 1), 'white')
pixels = img.load()

for i in range(-size, size + 1):
    for j in range(-size, size + 1):
        pixels[j+size, i+size] = (65, 105, 225) if get_res(i, j) == 0 else (255, 215, 0)

img.save('grid_visualization.png')