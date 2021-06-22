import logging  # Log everything for debugging purposes.
import numpy as np

from apocrita_deeprhea.deep_rhea.Game import Game

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

    class RHEAIndividual:
        """
        Each RHEA Individual handles their operations themselves and reports
        to RHEAPopulation.
        """

        def __init__(self, game: Game, args, board, parent1=None, parent2=None):
            self.INDIVIDUAL_LENGTH = args.INDIVIDUAL_LENGTH  # How long is the action plan.
            self.MUTATION_STRENGTH = args.MUTATION_STRENGTH  # How many mutations after crossover.
            self.parent1 = parent1                           # If exists, the first associated parent.
            self.parent2 = parent2                           # If exists, the second parent.
            self.action_plan = None                          # The action plan for the individual.
            self.fitness = None                              # Fitness of the individual.
            self.game = game                                 # Game information relayed to individual.
            self.board = board

            # Set the action plan:
            if parent1 is not None and parent2 is not None:
                self.build_from_parents()
            elif parent2 is None and parent2 is None:
                # If there are no parents: Construct a random sequence.
                self.build_from_scratch()
            else:
                raise ValueError("You need to input two parents or no parents!")

            # Execute the action plan and learn the fitness:
            self.measure_fitness()

        def build_from_parents(self):
            """
            From the current state of the game, build a VALID action plan of given
            individual length and parents. Mutations must return a valid action plan.

            :return: Returns a valid action plan with uniform crossover and random mutation.
            """

            # Build boolean sequence: 0 for parent 1, 1 for parent 2.
            crossover_idx = np.random.randint(2, size=self.INDIVIDUAL_LENGTH)

            # Build the sequence: (Does uniform crossover)
            # If crossover index is zero; take the value from parent 1, else from parent 2 for all indices available.
            # ToDo: Implement different kind of crossovers. (?-Further Refinements)
            draft_plan = [self.parent1.action_plan[i] if crossover_idx[i] == 0
                          else self.parent2.action_plan[i] for i in crossover_idx]

            self.action_plan = draft_plan  # FixMe: For now ignore the mutations. Mutate into a valid sequence:

        # ToDo: Complete this:
        def measure_fitness(self):
            raise NotImplementedError

        # ToDo: Complete this:
        def build_from_scratch(self):
            raise NotImplementedError