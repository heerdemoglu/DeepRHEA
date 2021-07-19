import RHEAPopulation
from apocrita_deeprhea.othello.OthelloGame import OthelloGame
from apocrita_deeprhea.othello.pytorch.NNet import NNetWrapper
from apocrita_deeprhea.othello.OthelloLogic import Board
from utils import dotdict
import numpy as np
game = OthelloGame(n=6)
nnet = NNetWrapper(game)

# FixMe: Loading pre-trained model gives almost 1 fitness (implying player 1 will certainly win)
nnet.load_checkpoint(r'C:\Users\heerd\PycharmProjects\DeepRHEA\trained_model', '6x6_153checkpoints_best.pth.tar')
# nnet.load_checkpoint(r'C:\Users\heerd\PycharmProjects\DeepRHEA\trained_model', '6x100x25_best.pth.tar')


args = dotdict({
    'NUM_OF_INDIVIDUALS': 50,
    'INDIVIDUAL_LENGTH': 5,
    'NUM_OF_BEST_INDIVIDUALS': 5,
    'MAX_GENERATION_BUDGET': 10,
    'MUTATION_CHANCE': 0.4,  # Number of complete self-play games to simulate during a new iteration.
})

# Creating sample board:
board = Board(6)

# Creating individuals: -- No meaning meant here, random creations.
population = RHEAPopulation.RHEAPopulation(game=game, nnet=nnet, args=args, board=board)
print(population.debug_print_population())

# Play 5 turns:
for i in range(5):
    print('Turn ', i)
    population.evolve()

    # Select the best individual and play it; proceed with the game tick:
    population.select_and_execute_individual()

