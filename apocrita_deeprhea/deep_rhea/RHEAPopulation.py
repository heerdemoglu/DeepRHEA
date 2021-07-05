import logging  # Log everything for debugging purposes.
import RHEAIndividual

from apocrita_deeprhea.othello.OthelloLogic import Board

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
        self.board = Board(6)

        # Construct the list of individuals and their fitness:
        self.individuals = []
        self.indv_fitness = []
        for i in range(self.args.NUM_OF_INDIVIDUALS):
            indv = RHEAIndividual.RHEAIndividual(game=game, args=args, nnet=nnet, board=self.board)
            self.indv_fitness.append(indv.get_fitness())
            self.individuals.append(indv)

        # FixMe: These sections will be migrated to evolve and execute methods of this class.
        # Sort the population with respect to their fitness: (descending order)
        sorted_population = self.sort_population_fitness()

        # Calculate the rankings, apply them to sorted population:
        total_fitness = sum(self.indv_fitness)

        for i in range(len(sorted_population)):  # each list element is a tuple with (fitness, individual)
            sorted_population[i][0] = sorted_population[i][0] / total_fitness * 100

        print('test')

        # Pick first elite individuals (NUM_OF_BEST_INDIVIDUALS) that will ascend to the next generation.
        elites = sorted_population[:self.args.NUM_OF_BEST_INDIVIDUALS]
        del sorted_population[:self.args.NUM_OF_BEST_INDIVIDUALS]

        # Pick rest of the individuals by Rank Selection, cross over and mutate.

        # Evolve the population while computational budget is not reached (different method -- search)

    def sort_population_fitness(self):
        list_ = list(zip(self.indv_fitness, self.individuals))  # list(), otherwise iterator ends and returns empty.
        listed = [list(a) for a in list_]
        sorted_individuals = sorted(listed, reverse=True)

        return sorted_individuals

    def crossover_parents(self):
        pass

    def evolve(self):
        pass
