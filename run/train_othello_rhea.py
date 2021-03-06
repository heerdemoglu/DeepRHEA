import logging
import coloredlogs
from torch.utils.tensorboard import SummaryWriter

from deep_rhea.Coach import Coach
from core_game.utils import dotdict
from othello.OthelloGame import OthelloGame
from othello.pytorch.NNet import NNetWrapper


def main():
    # Directory Setup:
    # Fetch and setup to Apocrita directory.
    CHK_DIR = "C:/Users/heerd/PycharmProjects/DeepRHEA/run/best_models"

    # Set the loggers:
    log = logging.getLogger(__name__)
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

    # Training arguments:
    args = dotdict({
        'numIters': 50,
        'numEps': 20,  # Number of complete self-play games to simulate during a new iteration.
        'updateThreshold': 0.55,

        # During arena playoff, new nnet will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
        'arenaCompare': 10,  # Number of games to play during arena play to determine if new net will be accepted.
        'checkpoint': CHK_DIR,
        'load_model': True,
        'load_folder_file': (CHK_DIR, 'rhea.pth.tar', ''),
        'numItersForTrainExamplesHistory': 20,

        'NUM_OF_INDIVIDUALS': 10,
        'INDIVIDUAL_LENGTH': 5,
        'NUM_OF_BEST_INDIVIDUALS': 0,
        'MAX_GENERATION_BUDGET': 20,
        'MUTATION_CHANCE': 0.7,  # Number of complete self-play games to simulate during a new iteration.
        'CROSSOVER_MUTATIONS': 3,  # must be less than number of individuals.
    })
    # tensorboard --logdir=C:\Users\heerd\PycharmProjects\DeepRHEA\run\runs

    # Create the game and the neural network:
    game = OthelloGame(n=6)
    writer = SummaryWriter(comment="Test1")
    nnet = NNetWrapper(game, writer)


    # Wrote the parameters here:

    # Either learn from checkpoints or start from scratch.
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(game, nnet, args, writer)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.load_train_examples()

    log.info('Starting training.')
    c.learn()


if __name__ == "__main__":
    main()
