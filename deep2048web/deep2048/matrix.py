from random import randint
from .vector import Vector
from .player import Player
from copy import deepcopy

class Matrix:
    def __init__(self, name, size=6, win = 1):
        self.size = size
        self.matrix = []
        self.player = Player(name)
        self.win = win
        self.create()

    def create(self):
        self.matrix = []
        for i in range(self.size):
            tmp = []
            for j in range(self.size):
                tmp.append(0)
            self.matrix.append(tmp)
    
    def reset(self):
        self.win = 1
        self.create()
        self.set_rnd_empty_case(2)
        self.set_rnd_empty_case(2)
        self.player = Player(self.player.name)

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
        if (nb >= 1000):
            return None
        return Vector(rndX, rndY)

    def set_rnd_empty_case(self, nb):
        vec = self.get_rnd_empty_case()
        if (vec == None):
            self.player.finish = True
        else:
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
                    self.player.score += self.matrix[i - 1][j]

    def move_up(self):
        matrixBefore = deepcopy(self)
        self.go_up()
        self.merge_up()
        self.go_up()
        self.new_case(matrixBefore)

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
                    self.player.score += self.matrix[i + 1][j]

    def move_down(self):
        matrixBefore = deepcopy(self)
        self.go_down()
        self.merge_down()
        self.go_down()
        self.new_case(matrixBefore)

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
                    self.player.score += self.matrix[i][j - 1]

    def move_left(self):
        matrixBefore = deepcopy(self)
        self.go_left()
        self.merge_left()
        self.go_left()
        self.new_case(matrixBefore)

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
                    self.player.score += self.matrix[i][j + 1]

    def move_right(self):
        matrixBefore = deepcopy(self)
        self.go_right()
        self.merge_right()
        self.go_right()
        self.new_case(matrixBefore)

    def new_case(self, matrixBefore):
        if self.matrix != matrixBefore.matrix:
            self.player.moves += 1
            nb = randint(0, 9)
            if (nb == 0):
                self.set_rnd_empty_case(4)
            else:
                self.set_rnd_empty_case(2)
    
    def test_loose(self):
        new_mat = deepcopy(self)
        new_mat.move_up()
        print("up", new_mat.matrix)
        new_mat.move_down()
        print("d", new_mat.matrix)
        new_mat.move_left()
        print("l", new_mat.matrix)
        new_mat.move_right()
        print("r", new_mat.matrix)
        if new_mat.matrix == self.matrix:
            print(new_mat.matrix, self.matrix)
            self.win = 0
            return 0
        return 1

    def random_move(self):
        nb = randint(0,3)
        if nb == 0:
            self.move_up()
        if nb == 1:
            self.move_left()
        if nb == 2:
            self.move_down()
        if nb == 3:
            self.move_right()

    def move_inp(self, direction):
        if direction == "up":
            self.move_up()
        elif direction == "down":
            self.move_down()
        elif direction == "left":
            self.move_left()
        elif direction == "right":
            self.move_right()
