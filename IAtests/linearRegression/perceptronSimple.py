import math
import random

class PerceptronSimple:
    def __init__(self, nb_input):
        self.w = []
        for i in range(nb_input):
            self.w.append(random.randint(-1, 1))
        self.b = random.randint(-1, 1)
        self.learning_rate = 0.01

    def forward_propagation(self, input):
        y = 0
        for i in range(len(self.w)):
            y += input[i] * self.w[i]
        y += self.b
        return y

    def error(self, target, predicted):
        return (target - predicted)**2

    def minimization(self, input, predicted, target):
        error_derivative = 2 * (predicted - target)
        for i in range(len(self.w)):
            self.w[i] = self.w[i] - self.learning_rate * error_derivative * input[i]
        self.b = self.b - self.learning_rate * error_derivative
