import math
import random

class NetworkSimple:
    def __init__(self, nb_input, nb_output, width, length):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        self.learning_rate = 0.01

    def forward_propagation(self, input):
        y = 0
        for i in range(len(self.w)):
            y += input[i] * self.w[i]
        y += self.b
        return self.activation(y)

    def activation(self, x):
        return 1 / (1 + math.exp(-x))

    def error(self, target, predicted):
        return target - predicted

    def minimization(self, input, predicted, target):
        error = self.error(target, predicted)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + self.learning_rate * error * input[i]
        self.b = self.b + self.learning_rate * error
