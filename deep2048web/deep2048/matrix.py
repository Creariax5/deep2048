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

    def go_down(self):
        for i in range(self.size - 1, -1, -1):
            for j in range(self.size - 1, -1 ,-1):
                if self.matrix[i][j] != 0:
                    k = i
                    while k < self.size -1 and self.matrix[k + 1][j] == 0:
                        temp = self.matrix[k][j]
                        self.matrix[k][j] = self.matrix[k + 1][j]
                        self.matrix[k + 1][j] = temp
                        k += 1

    def merge_down(self):
        for i in range(self.size - 2, -1, -1):
            for j in range(self.size - 1, -1 ,-1):
                if self.matrix[i][j] == self.matrix[i + 1][j] and i < self.size -1:
                    self.matrix[i][j] = 0
                    self.matrix[i + 1][j] *= 2

    def move_down(self):
        self.go_down()
        self.merge_down()
        self.go_down()

    def go_left(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != 0:
                    k = j
                    while k > 0 and self.matrix[i][k - 1] == 0:
                        temp = self.matrix[i][k]
                        self.matrix[i][k] = self.matrix[i][k - 1]
                        self.matrix[i][k - 1] = temp
                        k -= 1

    def merge_left(self):
        for i in range(self.size):
            for j in range(1, self.size):
                if self.matrix[i][j] == self.matrix[i][j - 1] and j > 0:
                    self.matrix[i][j] = 0
                    self.matrix[i][j - 1] *= 2

    def move_left(self):
        self.go_left()
        self.merge_left()
        self.go_left()

    def go_right(self):
        for i in range(self.size - 1, -1, -1):
            for j in range(self.size - 1, -1 ,-1):
                if self.matrix[i][j] != 0:
                    k = j
                    while k < self.size -1 and self.matrix[i][k + 1] == 0:
                        temp = self.matrix[i][k]
                        self.matrix[i][k] = self.matrix[i][k + 1]
                        self.matrix[i][k + 1] = temp
                        k += 1

    def merge_right(self):
        for i in range(self.size - 1, -1, -1):
            for j in range(self.size - 2, -1 ,-1):
                if self.matrix[i][j] == self.matrix[i][j + 1] and j < self.size -1:
                    self.matrix[i][j] = 0
                    self.matrix[i][j + 1] *= 2

    def move_right(self):
        self.go_right()
        self.merge_right()
        self.go_right()

    def move_inp(self):
        inp = input()
        if inp == "up":
            self.move_up()
        elif inp == "down":
            self.move_down()
        elif inp == "left":
            self.move_left()
        elif inp == "right":
            self.move_right()
        else:
            self.move_inp()
            return
