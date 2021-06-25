import logging  # Log everything for debugging purposes.
import numpy as np

log = logging.getLogger(__name__)


class RHEAPopulation:
    """
    RHEA population maintains a population of individuals and evolves them
    for generations. It also utilizes shift-buffer to carry search from past
    generations to current generations.
    """

    # From MCTS implementation must use.
    def __init__(self, game, nnet, args):
        """
        Initializes the RHE Search with a population of N;
        evolves the best individual and passes it to the rest of the
        code.

        :param game:
        :param nnet:
        :param args:
        """
        self.game = game
        self.nnet = nnet
        # {num_of_individuals, individual_length, num_of_best_individuals}
        self.args = args

    # From MCTS implementation must use.
    def get_action_prob(self, canonical_board, temp=1):
        raise NotImplementedError

    # From MCTS implementation must use.
    def search(self, canonical_board):
        raise NotImplementedError

    def update_population_boards(self, canonical_board):
        """
        Updates all parents of the population with the new state of the board for future game ticks.
        :param canonical_board:
        :return:
        """
        raise NotImplementedError

