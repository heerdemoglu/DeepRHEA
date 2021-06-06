# import os
import logging
import coloredlogs
from Coach import Coach
from utils import dotdict
from othello.pytorch.NNet import NNetWrapper
from othello.OthelloGame import OthelloGame

# Directory Setup:
ROOT_DIR = r"C:\Users\heerd\PycharmProjects\DeepRHEA"
CHK_DIR = r".\temp"


# Set the loggers:
log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# Training arguments:
args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new nnet will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': CHK_DIR,
    'load_model': False,
    'numItersForTrainExamplesHistory': 20,
})

# This is for debugging purposes; actual model has to run for a long time:
args['numIters'] = 10
args['numEps'] = 20
args['arenaCompare'] = 5

game = OthelloGame(n=6)
nnet = NNetWrapper(game)

coach = Coach(game, nnet, args)
coach.learn()
