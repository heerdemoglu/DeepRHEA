from torch.utils.tensorboard import SummaryWriter

from alpha_zero.MCTS import MCTS
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

# Only one should be true: (If all false compete with MCTS)
human_vs_cpu = False
random_vs_cpu = False
greedy_vs_cpu = True
rhea_vs_rhea = False

print('VS HUMAN: ', human_vs_cpu)
print('VS random: ', random_vs_cpu)
print('VS greedy: ', greedy_vs_cpu)
print('VS rhea: ', rhea_vs_rhea)


if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

checkpoint_dir = r'C:\Users\heerd\PycharmProjects\DeepRHEA\run\best_models'
writer = SummaryWriter(log_dir='pit', comment='rhea_run1')

# nnet players
n1 = NNet(g, writer=writer)
if mini_othello:
    n1.load_checkpoint(checkpoint_dir, 'rhea.pth.tar')
else:
    n1.load_checkpoint(checkpoint_dir, 'rhea8.pth.tar')

args1 = dotdict({'NUM_OF_INDIVIDUALS': 15,
                     'INDIVIDUAL_LENGTH': 5,
                     'NUM_OF_BEST_INDIVIDUALS': 0,
                     'MAX_GENERATION_BUDGET': 25,
                     'MUTATION_CHANCE': 0.7,  # Number of complete self-play games to simulate during a new iteration.
                     'CROSSOVER_MUTATIONS': 3,  # must be less than number of individuals.
                     })
# mcts1 = MCTS(g, n1, args1)

rhea = RHEAPopulation(game=g, nnet=n1, args=args1, board=Board(6))
action1 = rhea.evolve()

if human_vs_cpu:
    player2 = hp
elif random_vs_cpu:
    player2 = rp
elif greedy_vs_cpu:
    player2 = gp
elif rhea_vs_rhea:
    n2 = NNet(g, writer)
    n2.load_checkpoint(checkpoint_dir, 'rhea.pth.tar')
    args2 = dotdict({'NUM_OF_INDIVIDUALS': 20,
                     'INDIVIDUAL_LENGTH': 5,
                     'NUM_OF_BEST_INDIVIDUALS': 0,
                     'MAX_GENERATION_BUDGET': 10,
                     'MUTATION_CHANCE': 0.7,  # Number of complete self-play games to simulate during a new iteration.
                     'CROSSOVER_MUTATIONS': 3,  # must be less than number of individuals.
                     })
    player2 = RHEAPopulation(game=g, nnet=n2, args=args2, player=-1, board=Board(6))
else:
    n2 = NNet(g, writer)
    n2.load_checkpoint(checkpoint_dir, 'mcts.pth.tar')
    args2 = dotdict({'numMCTSSims': 20, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    player2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena(rhea, player2, g)
one_won, two_won, draw = arena.playGames(10, verbose=True)
print('One won: ', one_won, ' Two won: ', two_won, 'Draw: ', draw)
