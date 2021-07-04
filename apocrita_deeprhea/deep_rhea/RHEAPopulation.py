import logging  # Log everything for debugging purposes.
import RHEAIndividual

log = logging.getLogger(__name__)


class RHEAPopulation:
    """
    RHEA population maintains a population of individuals and evolves them
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
        # {num_of_individuals, individual_length, num_of_best_individuals, mutation_chance, max_generation_budget}
        self.args = args
        self.current_player = 1  # Always start with player 1. (design choice)

        # Get the initial board configuration to set up the individuals.
        board = self.game.getInitBoard()

        # Construct the list of individuals and their fitness:
        self.individuals = []
        self.indv_fitness = []
        for i in range(self.args.NUM_OF_INDIVIDUALS):
            indv = RHEAIndividual.RHEAIndividual(game=game, args=args, nnet=nnet, board=board)
            self.indv_fitness = indv.get_fitness()
            self.individuals.append(indv)

    def sort_population_fitness(self):
        pass

    def pick_best_individuals(self):
        pass

    def crossover_parents(self):
        pass

    def executeGeneration(self):
        pass