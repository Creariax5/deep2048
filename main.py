from matrix import Matrix
from player import Player

def start_board(matrix):
    vec = matrix.get_rnd_empty_case()
    matrix.matrix[vec.x][vec.y] = 2
    vec = matrix.get_rnd_empty_case()
    matrix.matrix[vec.x][vec.y] = 2
    matrix.display()

def game_loop(matrix, player):
    while not player.finish:
        inp = input()
        print()
        matrix.move(inp)
        matrix.set_rnd_empty_case(2)
        matrix.display()

player = Player("Jeremy")
matrix = Matrix()

start_board(matrix)
game_loop(matrix, player)
