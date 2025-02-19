import random
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from perceptronSimple import PerceptronSimple


digits = load_digits()
print(digits.data.shape)
plt.gray()
for i in range(1):
    plt.matshow(digits.images[i])
    plt.show()

neuron = PerceptronSimple(2)
