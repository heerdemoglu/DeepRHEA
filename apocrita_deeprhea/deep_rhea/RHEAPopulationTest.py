import RHEAPopulation
from apocrita_deeprhea.othello.OthelloGame import OthelloGame
from apocrita_deeprhea.othello.OthelloLogic import Board
from apocrita_deeprhea.othello.pytorch.NNet import NNetWrapper
from utils import dotdict

game = OthelloGame(n=6)
nnet = NNetWrapper(game)

nnet.load_checkpoint(r'C:\Users\heerd\PycharmProjects\DeepRHEA\trained_model', '6x6_153checkpoints_best.pth.tar')
# nnet.load_checkpoint(r'C:\Users\heerd\PycharmProjects\DeepRHEA\trained_model', '6x100x25_best.pth.tar')


args = dotdict({
    'NUM_OF_INDIVIDUALS': 25,
    'INDIVIDUAL_LENGTH': 5,
    'NUM_OF_BEST_INDIVIDUALS': 5,
    'MAX_GENERATION_BUDGET': 10,
    'MUTATION_CHANCE': 0.2,  # Number of complete self-play games to simulate during a new iteration.
})

# Creating sample board:
board = Board(6)

# Creating individuals: -- No meaning meant here, random creations.
population = RHEAPopulation.RHEAPopulation(game=game, nnet=nnet, args=args, board=board)

# Play 10 turns: - Why so slow? Optimize!
for i in range(5):
    print('Turn ', i+1)
    # Evolve 10 iterations:
    for j in range(5):
        print('Generation ', j+1, ' computed.')
        population.evolve()

    # Select the best individual and play it; proceed with the game tick:
    population.select_and_execute_individual()
