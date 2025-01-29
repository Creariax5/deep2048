
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt

from visuData import visualizing as visu
from perceptron.perceptron1957 import Perceptron1957

# Reading the data
data_set = pd.read_csv('./dataset/used_car_dataset.csv')

print(data_set)

# Dropping the id column
data_set.drop('Brand', axis=1, inplace=True)
data_set.drop('model', axis=1, inplace=True)
data_set.drop('Transmission', axis=1, inplace=True)
data_set.drop('PostedDate', axis=1, inplace=True)
data_set.drop('AdditionInfo', axis=1, inplace=True)
#data_set.drop('Transmission', axis=1, inplace=True)
print(data_set)
owner = {
    'first': 0,
    'second': 1
}

# Replace values using map
data_set['Species'] = data_set['Owner'].map(owner)

#visu.plot_data(data_set)
