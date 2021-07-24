import sys
import logging
import coloredlogs

# todo: change for DeepRHEA
from deep_rhea.Coach import Coach
from core_game.utils import dotdict
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper


def main(home):
    # Directory Setup:
    # Fetch and setup to Apocrita directory.
    HOME_DIR = home
    CHK_DIR = "models/checkpoint"

    # Set the loggers:
    log = logging.getLogger(__name__)
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

    # Training arguments:
    args = dotdict({
        'numIters': 1000,
        'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,  #
        'updateThreshold': 0.6,
        # During arena playoff, new nnet will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
        'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,
        'checkpoint': CHK_DIR,
        'load_model': False,
        'load_folder_file': (CHK_DIR, 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        'NUM_OF_INDIVIDUALS': 10,
        'INDIVIDUAL_LENGTH': 3,
        'NUM_OF_BEST_INDIVIDUALS': 2,
        'MAX_GENERATION_BUDGET': 20,
        'MUTATION_CHANCE': 0.8,  # Number of complete self-play games to simulate during a new iteration.
        'CROSSOVER_MUTATIONS': 3,  # must be less than number of individuals.
        'REWARD_DECAY_RATE': 0.9,  # how the further states decade the fitness.
    })

    # # This is for debugging purposes; actual model has to run for a long time:
    # args['numIters'] = 100
    # args['numEps'] = 50
    # args['arenaCompare'] = 20

    # Create the game and the neural network:
    game = OthelloGame(n=6)
    nnet = NNetWrapper(game)

    # Either learn from checkpoints or start from scratch.
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(game, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting training.')
    c.learn()


if __name__ == "__main__":
    directory = sys.argv[0]
    main(directory)