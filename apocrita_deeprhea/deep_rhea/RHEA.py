import logging  # Log everything for debugging purposes.
import numpy as np

from apocrita_deeprhea.deep_rhea.Game import Game

log = logging.getLogger(__name__)


class RHEAPopulation:
    """ RHEA population maintains a population of individuals and evolves them
    for generations. It also utilizes shift-buffer to carry search from past
    generations to current generations.
    """

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

    def get_action_prob(self, canonical_board, temp=1):
        raise NotImplementedError

    def search(self, canonical_board):
        raise NotImplementedError

    class RHEAIndividual:
        """
        Each RHEA Individual handles their operations themselves and reports
        to RHEAPopulation.
        """

        def __init__(self, game: Game, args, parent1=None, parent2=None):
            self.INDIVIDUAL_LENGTH = args.INDIVIDUAL_LENGTH  # How long is the action plan.
            self.MUTATION_STRENGTH = args.MUTATION_STRENGTH  # How many mutations after crossover.
            self.action_plan = None  # The action plan for the individual.
            self.fitness = None  # Fitness of the individual.
            self.game = game  # Game information relayed to individual.

            # Set the action plan:
            if parent1 is not None and parent2 is not None:
                self.build_from_parents(self, parent1, parent2)  # ToDo: Fix unexpected argument 'parent2'. why?
            elif parent1 is None and parent2 is None:
                # ToDo: Complete this:
                self.action_plan = None
            else:
                raise ValueError("You need to input two parents!")

            # ToDo: Execute the action plan and learn the fitness:

        # ToDo: Complete this.
        def build_from_parents(self, parent1, parent2):
            """
            From the current state of the game, build a VALID action plan of given
            individual length and parents. Mutations must return a valid action plan.

            :param parent1:
            :param parent2:
            :return:
            """

            # Build boolean sequence: 0 for parent 1, 1 for parent 2.
            crossover_idx = np.random.randint(2, size=self.INDIVIDUAL_LENGTH)

            # Build the sequence: (Does uniform crossover)
            # ToDo: Implement different kind of crossovers. (?)
            draft_plan = [parent1[i] if crossover_idx[i] == 0 else parent2[i] for i in crossover_idx]

            # ToDo: Mutate into a valid sequence:
            # self.game.getValidMoves(self, canonical_board, player)
            pass
