from random import randint
from .vector import Vector
from .player import Player, Model
from copy import deepcopy
from typing import List, Tuple

class MoveRecord:
    def __init__(self, value: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int], merged: bool = False):
        self.value = value
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.merged = merged

    def __str__(self):
        action = "merged into" if self.merged else "moved to"
        return f"Block {self.value} {action} ({self.to_pos[0]}, {self.to_pos[1]}) from ({self.from_pos[0]}, {self.from_pos[1]})"

class Matrix:
    def __init__(self, name, model=Model.HUMAN, size=4, win=1):
        self.size = size
        self.matrix = []
        self.player = Player(name, model)
        self.win = win
        self.playing = False
        self.move_history: List[MoveRecord] = []
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
        self.player = Player(self.player.name, self.player.model)

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

    def go_up(self, human=False):
        moves = [] if human else None
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != 0:
                    k = i
                    while k > 0 and self.matrix[k - 1][j] == 0:
                        if human:
                            moves.append(MoveRecord(
                                self.matrix[k][j],
                                (k, j),
                                (k - 1, j)
                            ))
                        temp = self.matrix[k][j]
                        self.matrix[k][j] = self.matrix[k - 1][j]
                        self.matrix[k - 1][j] = temp
                        k -= 1
        return moves

    def merge_up(self, human=False):
        moves = [] if human else None
        for i in range(1, self.size):
            for j in range(self.size):
                if self.matrix[i][j] == self.matrix[i - 1][j] and i > 0:
                    if human:
                        moves.append(MoveRecord(
                            self.matrix[i][j],
                            (i, j),
                            (i - 1, j),
                            merged=True
                        ))
                    self.matrix[i][j] = 0
                    self.matrix[i - 1][j] *= 2
                    self.player.score += self.matrix[i - 1][j]
        return moves

    def move_up(self, human=False):
        matrixBefore = deepcopy(self)
        self.move_history = []  # Clear previous move history
        
        if human:
            moves1 = self.go_up(human=True)
            moves2 = self.merge_up(human=True)
            moves3 = self.go_up(human=True)
            self.move_history.extend(moves1)
            self.move_history.extend(moves2)
            self.move_history.extend(moves3)
        else:
            self.go_up()
            self.merge_up()
            self.go_up()
            
        self.new_case(matrixBefore)

    def go_down(self, human=False):
        moves = [] if human else None
        for i in range(self.size - 1, -1, -1):
            for j in range(self.size - 1, -1 ,-1):
                if self.matrix[i][j] != 0:
                    k = i
                    while k < self.size - 1 and self.matrix[k + 1][j] == 0:
                        if human:
                            moves.append(MoveRecord(
                                self.matrix[k][j],
                                (k, j),
                                (k + 1, j)
                            ))
                        temp = self.matrix[k][j]
                        self.matrix[k][j] = self.matrix[k + 1][j]
                        self.matrix[k + 1][j] = temp
                        k += 1
        return moves

    def merge_down(self, human=False):
        moves = [] if human else None
        for i in range(self.size - 2, -1, -1):
            for j in range(self.size - 1, -1 ,-1):
                if self.matrix[i][j] == self.matrix[i + 1][j] and i < self.size - 1:
                    if human:
                        moves.append(MoveRecord(
                            self.matrix[i][j],
                            (i, j),
                            (i + 1, j),
                            merged=True
                        ))
                    self.matrix[i][j] = 0
                    self.matrix[i + 1][j] *= 2
                    self.player.score += self.matrix[i + 1][j]
        return moves

    def move_down(self, human=False):
        matrixBefore = deepcopy(self)
        self.move_history = []  # Clear previous move history
        
        if human:
            moves1 = self.go_down(human=True)
            moves2 = self.merge_down(human=True)
            moves3 = self.go_down(human=True)
            self.move_history.extend(moves1)
            self.move_history.extend(moves2)
            self.move_history.extend(moves3)
        else:
            self.go_down()
            self.merge_down()
            self.go_down()
            
        self.new_case(matrixBefore)

    def go_left(self, human=False):
        moves = [] if human else None
        for i in range(self.size):
            for j in range(self.size):
                if self.matrix[i][j] != 0:
                    k = j
                    while k > 0 and self.matrix[i][k - 1] == 0:
                        if human:
                            moves.append(MoveRecord(
                                self.matrix[i][k],
                                (i, k),
                                (i, k - 1)
                            ))
                        temp = self.matrix[i][k]
                        self.matrix[i][k] = self.matrix[i][k - 1]
                        self.matrix[i][k - 1] = temp
                        k -= 1
        return moves

    def merge_left(self, human=False):
        moves = [] if human else None
        for i in range(self.size):
            for j in range(1, self.size):
                if self.matrix[i][j] == self.matrix[i][j - 1] and j > 0:
                    if human:
                        moves.append(MoveRecord(
                            self.matrix[i][j],
                            (i, j),
                            (i, j - 1),
                            merged=True
                        ))
                    self.matrix[i][j] = 0
                    self.matrix[i][j - 1] *= 2
                    self.player.score += self.matrix[i][j - 1]
        return moves

    def move_left(self, human=False):
        matrixBefore = deepcopy(self)
        self.move_history = []  # Clear previous move history
        
        if human:
            moves1 = self.go_left(human=True)
            moves2 = self.merge_left(human=True)
            moves3 = self.go_left(human=True)
            self.move_history.extend(moves1)
            self.move_history.extend(moves2)
            self.move_history.extend(moves3)
        else:
            self.go_left()
            self.merge_left()
            self.go_left()
            
        self.new_case(matrixBefore)

    def go_right(self, human=False):
        moves = [] if human else None
        for i in range(self.size - 1, -1, -1):
            for j in range(self.size - 1, -1 ,-1):
                if self.matrix[i][j] != 0:
                    k = j
                    while k < self.size - 1 and self.matrix[i][k + 1] == 0:
                        if human:
                            moves.append(MoveRecord(
                                self.matrix[i][k],
                                (i, k),
                                (i, k + 1)
                            ))
                        temp = self.matrix[i][k]
                        self.matrix[i][k] = self.matrix[i][k + 1]
                        self.matrix[i][k + 1] = temp
                        k += 1
        return moves

    def merge_right(self, human=False):
        moves = [] if human else None
        for i in range(self.size - 1, -1, -1):
            for j in range(self.size - 2, -1 ,-1):
                if self.matrix[i][j] == self.matrix[i][j + 1] and j < self.size - 1:
                    if human:
                        moves.append(MoveRecord(
                            self.matrix[i][j],
                            (i, j),
                            (i, j + 1),
                            merged=True
                        ))
                    self.matrix[i][j] = 0
                    self.matrix[i][j + 1] *= 2
                    self.player.score += self.matrix[i][j + 1]
        return moves

    def move_right(self, human=False):
        matrixBefore = deepcopy(self)
        self.move_history = []  # Clear previous move history
        
        if human:
            moves1 = self.go_right(human=True)
            moves2 = self.merge_right(human=True)
            moves3 = self.go_right(human=True)
            self.move_history.extend(moves1)
            self.move_history.extend(moves2)
            self.move_history.extend(moves3)
        else:
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
        #print("up", new_mat.matrix)
        new_mat.move_down()
        #print("d", new_mat.matrix)
        new_mat.move_left()
        #print("l", new_mat.matrix)
        new_mat.move_right()
        #print("r", new_mat.matrix)
        if new_mat.matrix == self.matrix:
            #print(new_mat.matrix, self.matrix)
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

    def move_inp(self, direction, human=False):
        if direction == "up":
            self.move_up(human)
        elif direction == "down":
            self.move_down(human)
        elif direction == "left":
            self.move_left(human)
        elif direction == "right":
            self.move_right(human)
            
        if human:
            return self.move_history

    def get_move_history(self):
        return self.move_history

    def print_move_history(self):
        print("\nMove History:")
        for move in self.move_history:
            print(str(move))
