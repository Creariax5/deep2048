import math
<<<<<<< HEAD


class PerceptronSimple:
    def __init__(self, input, b):
        self.w = input
        self.b = b
=======
import random

class PerceptronSimple:
    def __init__(self, nb_input):
        self.w = []
        for i in range(nb_input):
            self.w.append(random.randint(-1, 1))
        self.b = random.randint(-1, 1)
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5
        self.learning_rate = 0.01
    
    def forward_propagation(self, input):
        y = 0
        for i in range(len(self.w)):
            y += input[i] * self.w[i]
        y += self.b
        return self.activation(y)

    def activation(self, x):
        return 1 / (1 + math.exp(-x))

<<<<<<< HEAD
    def minimization(self, input, predicted, target):
        error = target - predicted
=======
    def error(self, target, predicted):
        return target - predicted

    def minimization(self, input, predicted, target):
        error = self.error(target, predicted)
>>>>>>> b96ec48a0387761af555a9dc5c1398a2b10a6cb5
        for i in range(len(self.w)):
            self.w[i] = self.w[i] + self.learning_rate * error * input[i]
        self.b = self.b + self.learning_rate * error
