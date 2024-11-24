import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from IAtests.visuData import visualizing as visu
from IAtests.perceptron.perceptron1957 import Perceptron1957

# Reading the data
iris = pd.read_csv('../dataset/Iris.csv')

# Dropping the id column
iris.drop('Id', axis=1, inplace=True)
iris.drop('SepalLengthCm', axis=1, inplace=True)
iris.drop('SepalWidthCm', axis=1, inplace=True)

species_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 1
}

# Replace values using map
iris['Species'] = iris['Species'].map(species_map)

visu.plot_data(iris)


def train(iris, neurone):
    train_set = iris.iloc[:-9]

    for i in range(len(train_set)):
        row = train_set.iloc[i]
        x, y, type = row.values[:3]
        retunVal = neurone.run(x, y)
        neurone.learning(x, y, retunVal, type)

def evaluate(iris, neurone):
        correct = 0
        test_set = iris.iloc[-9:]

        for i in range(len(test_set)):
            row = test_set.iloc[i]
            x, y, type = row.values[:3]
            retunVal = neurone.run(x, y)
            if retunVal == type:
                correct += 1

        accuracy = correct / len(test_set)
        return accuracy

def learning_speed_test(learning_rate):
        neurone = Perceptron1957()
        neurone.learning_rate = learning_rate
        last_accuracy = 0
        accuracy = 0
        i = 0

        while accuracy <= 0.9:
            train(iris, neurone)
            last_accuracy = evaluate(iris, neurone)
            if accuracy != last_accuracy:
                accuracy = last_accuracy
                print(f"Accuracy: {accuracy:.2%} in ", i)
            i += 1

def learning_rate_test(begin, laps, multi):
        learning_rate = begin
        for i in range(laps):
            print(f"learning rate: {learning_rate:.4f}")
            learning_speed_test(learning_rate)
            print()

            learning_rate = learning_rate * multi

#learning_rate_test(10, 40, 0.8)

learning_speed_test(0.01)
