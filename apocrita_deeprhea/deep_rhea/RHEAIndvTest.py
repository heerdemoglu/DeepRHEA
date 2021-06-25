import RHEAIndividual
from apocrita_deeprhea.othello.OthelloGame import OthelloGame
from apocrita_deeprhea.othello.pytorch.NNet import NNetWrapper
from apocrita_deeprhea.othello.OthelloLogic import Board
from utils import dotdict

game = OthelloGame(n=6)
nnet = NNetWrapper(game)

args = dotdict({
        'INDIVIDUAL_LENGTH': 5,
        'MUTATION_STRENGTH': 1,  # Number of complete self-play games to simulate during a new iteration.
})

# Creating sample board:
board = Board(6)

# Creating individuals
indv_1 = RHEAIndividual.RHEAIndividual(game=game, args=args, board=board, parent1=None, parent2=None)
indv_2 = RHEAIndividual.RHEAIndividual(game=game, args=args, board=board, parent1=None, parent2=None)
indv_3 = RHEAIndividual.RHEAIndividual(game=game, args=args, board=board, parent1=indv_1, parent2=indv_2)

print('Indv 1: ', indv_1.get_gene())
print('Indv 2: ', indv_2.get_gene())
print('Indv 3: ', indv_3.get_gene())
