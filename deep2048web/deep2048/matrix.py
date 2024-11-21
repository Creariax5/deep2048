from random import randint
from .vector import Vector


class Matrix:
    def __init__(self, size=6):
        self.size = size
        self.matrix = []
        self.create()

    def create(self):
        self.matrix = []
        for i in range(self.size):
            tmp = []
            for j in range(self.size):
                tmp.append(0)
            self.matrix.append(tmp)
    
    def reset(self):
        self.create()
        self.set_rnd_empty_case(2)

    def display(self):
        for i in range(self.size):
            print(self.matrix[i])

    def get_rnd_empty_case(self):
        nb = 0
        rndX = randint(0, self.size - 1)
        rndY = randint(0, self.size - 1)
        while self.matrix[rndX][rndY] != 0 and nb < 1000:
            rndX = randint(0, self.size - 1)
            rndY = randint(0, self.size - 1)
            nb += 1
        return Vector(rndX, rndY)

    def set_rnd_empty_case(self, nb):
        vec = self.get_rnd_empty_case()
        self.matrix[vec.x][vec.y] = nb

    def go_up(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != 0:
                    k = i
                    while k > 0 and self.matrix[k - 1][j] == 0:
                        temp = self.matrix[k][j]
                        self.matrix[k][j] = self.matrix[k - 1][j]
                        self.matrix[k - 1][j] = temp
                        k -= 1

    def merge_up(self):
        for i in range(1, self.size):
            for j in range(self.size):
                if self.matrix[i][j] == self.matrix[i - 1][j] and i > 0:
                    self.matrix[i][j] = 0
                    self.matrix[i - 1][j] *= 2

    def move_up(self):
        self.go_up()
        self.merge_up()
        self.go_up()

    def move_inp(self):
        inp = input()
        if inp == "z":
            self.move_up()
        elif inp == "s":
            self.vec = Vector(-1, 0)
        elif inp == "q":
            self.vec = Vector(0, 1)
        elif inp == "d":
            self.vec = Vector(0, -1)
        else:
            self.move_inp()
            return
