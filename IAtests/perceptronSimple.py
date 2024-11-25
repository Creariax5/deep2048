import math


class PerceptronSimple:
    def __init__(self, input, b):
        self.w = input
        self.b = b
        self.learning_rate = 0.01
    
    def forward_propagation(self, input):
        y = 0
        for i in range(len(self.w)):
            y += input[i] * self.w[i]
        y += self.b
        return self.activation(y)

    def activation(self, x):
        if x < -709:
            return 0
        if x > 709:
            return 1
        return 1 / (1 + math.exp(-x))

    def minimization(self, input, predicted, target):
        for i in range(len(self.input)):
            self.w[1] = self.w[1] + self.learning_rate * (target - predicted) * input[1]
        self.b = self.b + self.learning_rate * (target - predicted)
