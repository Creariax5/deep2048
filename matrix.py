from random import randint
from vector import Vector


class Matrix:
    def __init__(self, size=6):
        self.size = size
        self.matrix = []
        self.create()

    def create(self):
        for i in range(self.size):
            tmp = []
            for j in range(self.size):
                tmp.append(0)
            self.matrix.append(tmp)

    def display(self):
        for i in range(self.size):
            print(self.matrix[i])

    def get_rnd_empty_case(self):
        rndX = randint(0, self.size - 1)
        rndY = randint(0, self.size - 1)
        while self.matrix[rndX][rndY] != 0:
            rndX = randint(0, self.size - 1)
            rndY = randint(0, self.size - 1)
        return Vector(rndX, rndY)

    def move(self, inp):
        pass
