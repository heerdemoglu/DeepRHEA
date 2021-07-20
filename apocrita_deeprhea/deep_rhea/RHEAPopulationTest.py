import RHEAPopulation
from apocrita_deeprhea.othello.OthelloGame import OthelloGame
from apocrita_deeprhea.othello.OthelloLogic import Board
from apocrita_deeprhea.othello.pytorch.NNet import NNetWrapper
from utils import dotdict

game = OthelloGame(n=6)
nnet = NNetWrapper(game)

nnet.load_checkpoint(r'C:\Users\heerd\PycharmProjects\DeepRHEA\trained_model', '6x6_153checkpoints_best.pth.tar')

args = dotdict({
    'NUM_OF_INDIVIDUALS': 10,
    'INDIVIDUAL_LENGTH': 10,
    'NUM_OF_BEST_INDIVIDUALS': 2,
    'MAX_GENERATION_BUDGET': 30,
    'MUTATION_CHANCE': 0.3,  # Number of complete self-play games to simulate during a new iteration.
    'REWARD_DECAY_RATE': 0.9,  # how the further states decade the fitness.
})

# Creating sample board:
board = Board(6)

# Creating individuals: -- No meaning meant here, random creations.
population = RHEAPopulation.RHEAPopulation(game=game, nnet=nnet, args=args, board=board)

# Play 10 turns: - Why so slow? Optimize!
for i in range(17):
    print('Turn ', i+1)
    population.evolve()

    # Select the best individual and play it; proceed with the game tick:
    population.select_and_execute_individual()
