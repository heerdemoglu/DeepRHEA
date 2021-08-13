from torch.utils.tensorboard import SummaryWriter

from alpha_zero.Arena import Arena
from alpha_zero.MCTS import MCTS
from core_game.utils import dotdict
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet
import numpy as np


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = True  # Play in 6x6 instead of the normal 8x8.

# Only one should be true: (If all false compete with MCTS)
human_vs_cpu = True
random_vs_cpu = False
greedy_vs_cpu = False

print('VS HUMAN: ', human_vs_cpu)
print('VS random: ', random_vs_cpu)
print('VS greedy: ', greedy_vs_cpu)

if mini_othello:
    g = OthelloGame(6)
else:
    g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

checkpoint_dir = r'C:\Users\heerd\PycharmProjects\DeepRHEA\run\best_models'
writer = SummaryWriter(log_dir='pit', comment='mcts_run1')
# nnet players
n1 = NNet(g, writer)
if mini_othello:
    n1.load_checkpoint(checkpoint_dir, 'mcts.pth.tar')
else:
    n1.load_checkpoint(checkpoint_dir, 'mcts8.pth.tar')
args1 = dotdict({'numMCTSSims': 20, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
elif random_vs_cpu:
    player2 = rp
elif greedy_vs_cpu:
    player2 = gp
else:
    n2 = NNet(g, writer)
    n2.load_checkpoint(checkpoint_dir, 'mcts.pth.tar')
    args2 = dotdict({'numMCTSSims': 20, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena(n1p, player2, g, display=OthelloGame.display)

print(arena.playGames(20, verbose=True))
