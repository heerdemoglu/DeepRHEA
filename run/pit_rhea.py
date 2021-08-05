from core_game.utils import dotdict
from deep_rhea.Arena import Arena
from deep_rhea.RHEAPopulation import RHEAPopulation
from othello.OthelloGame import OthelloGame
from othello.OthelloLogic import Board
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = True  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
if mini_othello:
    n1.load_checkpoint(r'C:\Users\heerd\PycharmProjects\DeepRHEA\run\models\checkpoint', 'best.pth.tar')
else:
    n1.load_checkpoint('run/models/checkpoint', 'best.pth.tar')

args1 = dotdict({'NUM_OF_INDIVIDUALS': 10,
                 'INDIVIDUAL_LENGTH': 5,
                 'NUM_OF_BEST_INDIVIDUALS': 2,
                 'MAX_GENERATION_BUDGET': 10,
                 'MUTATION_CHANCE': 0.5,  # Number of complete self-play games to simulate during a new iteration.
                 'CROSSOVER_MUTATIONS': 2,  # must be less than number of individuals.
                 })
# mcts1 = MCTS(g, n1, args1)

rhea = RHEAPopulation(game=g, nnet=n1, args=args1, board=Board(6))

action1 = rhea.evolve()

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('run/models/checkpoint', 'best.pth.tar')
    args2 = dotdict({'NUM_OF_INDIVIDUALS': 15,
                     'INDIVIDUAL_LENGTH': 3,
                     'NUM_OF_BEST_INDIVIDUALS': 2,
                     'MAX_GENERATION_BUDGET': 15,
                     'MUTATION_CHANCE': 0.8,  # Number of complete self-play games to simulate during a new iteration.
                     'CROSSOVER_MUTATIONS': 2,  # must be less than number of individuals.
                     })
    rhea2 = RHEAPopulation(game=g, nnet=n2, args=args2)
    n2p = lambda x: np.argmax(rhea2.getActionProb(x, temp=0))  # fixme

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

# arena = Arena(action1, player2, g, display=OthelloGame.display) 
arena = Arena(action1, player2, g)
print(arena.playGames(2, verbose=True))
