from network import Network
from game import Game
from utils import create_board

# game param
BOARD_SIZE = 4
ITER = 1
moves_list = ['up', 'left', 'down', 'right']

# network param
NB_INPUT = BOARD_SIZE * BOARD_SIZE
HIDDEN_LAYER_WIDTH = 64
HIDDEN_LAYER_LENGTH = 4
NB_OUTPUT = len(moves_list)
learning_rate=0.05

network = Network(NB_INPUT, HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_LENGTH, NB_OUTPUT, learning_rate)

empty_board = create_board(BOARD_SIZE)

for i in range(ITER):
    
    game = Game(empty_board.copy())
    game.add_new_tile(2)

    while not game.finished:
        move = network.process_move(game.board)
        game.move(moves_list[move])
