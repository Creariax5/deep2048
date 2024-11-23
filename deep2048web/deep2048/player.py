from enum import Enum

class Model(Enum):
    HUMAN = 0
    RANDOM = 1
    X = 2
    XX = 3
    XXX = 4

class Player:
    def __init__(self, name, model):
        self.name = name
        self.score = 0
        self.moves = 0
        self.finish = False
        self.model = model
