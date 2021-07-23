import deep_rhea.RHEAPopulation as RHEAPopulation
from othello.OthelloGame import OthelloGame
from othello.OthelloLogic import Board
from othello.pytorch.NNet import NNetWrapper
from core_game.utils import dotdict

game = OthelloGame(n=6)
nnet = NNetWrapper(game)

args = dotdict({
    'NUM_OF_INDIVIDUALS': 10,
    'INDIVIDUAL_LENGTH': 3,
    'NUM_OF_BEST_INDIVIDUALS': 2,
    'MAX_GENERATION_BUDGET': 20,
    'MUTATION_CHANCE': 0.4,  # Number of complete self-play games to simulate during a new iteration.
    'CROSSOVER_MUTATIONS': 3,  # must be less than number of individuals.
})

# Creating sample board:
board = Board(6)

# Creating individuals: -- No meaning meant here, random creations.
population = RHEAPopulation.RHEAPopulation(game=game, nnet=nnet, args=args, board=board)

# Play 10 turns: - Why so slow? Optimize!
for i in range(16):
    print('Turn ', i+1)
    population.evolve()

    game = population.game
    nnet = population.nnet
    board = population.board

    population = RHEAPopulation.RHEAPopulation(game=game, nnet=nnet, args=args, board=board)

    population.evolve()

    # Select the best individual and play it; proceed with the game tick:
    population.select_and_execute_individual()
