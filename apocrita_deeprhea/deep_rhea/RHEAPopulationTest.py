import RHEAPopulation
from apocrita_deeprhea.othello.OthelloGame import OthelloGame
from apocrita_deeprhea.othello.pytorch.NNet import NNetWrapper
from apocrita_deeprhea.othello.OthelloLogic import Board
from utils import dotdict

game = OthelloGame(n=6)
nnet = NNetWrapper(game)

args = dotdict({
    'NUM_OF_INDIVIDUALS': 50,
    'INDIVIDUAL_LENGTH': 3,
    'NUM_OF_BEST_INDIVIDUALS': 1,
    'MAX_GENERATION_BUDGET': 10,
    'MUTATION_CHANCE': 0.1,  # Number of complete self-play games to simulate during a new iteration.
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
    print(population.debug_print_population())

print(population.debug_print_population())
