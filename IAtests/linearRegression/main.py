import random
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt


import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from IAtests.visuData import visualizing as visu
from perceptronSimple import PerceptronSimple

# Reading the data
iris = pd.read_csv('../dataset/Iris.csv')

# Dropping the id column
iris.drop('Id', axis=1, inplace=True)
iris.drop('SepalLengthCm', axis=1, inplace=True)
iris.drop('SepalWidthCm', axis=1, inplace=True)
iris.drop('Species', axis=1, inplace=True)

visu.plot_data(iris)

def train(iris, neuron: PerceptronSimple):
    train_set = iris.iloc[:-9]

    for i in range(len(train_set)):
        row = train_set.iloc[i]
        x, y = row.values[:2]
        returnVal = neuron.forward_propagation([x])
        neuron.minimization([x], returnVal, y)
    return neuron

def evaluate(test_set, neurone):
    correct = 0
    threshold = 0.5

    for i in range(len(test_set)):
        row = test_set.iloc[i]
        x, y = row.values[:2]
        returnVal = neurone.forward_propagation([x])
        predicted = 1 if returnVal >= threshold else 0
        if predicted == y:
            correct += 1

    accuracy = correct / len(test_set)
    return accuracy

def learning_speed_test(neurone, learning_rate, MAX_ITER=100):
    neurone.learning_rate = learning_rate
    last_accuracy = 0
    accuracy = 0
    i = 0

    while accuracy <= 1 and MAX_ITER != i:
        neurone = train(iris, neurone)
        if i%20 == 0:
            # show('test.png', neurone)
            pass
        i += 1
    return neurone
        
def learning_rate_test(begin, laps, multi):
        learning_rate = begin
        for i in range(laps):
            print(f"learning rate: {learning_rate:.4f}")
            learning_speed_test(learning_rate)
            print()

            learning_rate = learning_rate * multi

def get_dataset():
    return iris

