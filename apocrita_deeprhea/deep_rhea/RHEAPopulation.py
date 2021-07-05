import logging  # Log everything for debugging purposes.
import random

import numpy as np

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
        self.sort_population_fitness()

        # Pick first elite individuals (NUM_OF_BEST_INDIVIDUALS) that will ascend to the next generation.
        elites = self.individuals[:self.args.NUM_OF_BEST_INDIVIDUALS]
        del self.individuals[:self.args.NUM_OF_BEST_INDIVIDUALS]
        del self.indv_fitness[:self.args.NUM_OF_BEST_INDIVIDUALS]

        # Calculate the rankings, apply them to sorted population:
        # See: https://stackoverflow.com/questions/34961489/rank-selection-in-ga
        remaining_indv = self.args.NUM_OF_INDIVIDUALS - self.args.NUM_OF_BEST_INDIVIDUALS
        total_fitness = remaining_indv * (remaining_indv + 1) / 2

        for i in range(len(self.individuals)):  # each list element is a tuple with (fitness, individual)
            self.individuals[i][0] = (
                                                 self.args.NUM_OF_INDIVIDUALS - self.args.NUM_OF_BEST_INDIVIDUALS - i) / total_fitness

        # Cumulative probabilities needed to pick individuals:
        self.cumulative_probabilities = []
        prob = 0
        for i in range(len(self.individuals)):
            prob += self.individuals[i][0]
            self.cumulative_probabilities.append(round(prob, 4))

        # todo: below.

        # Add elites to new population, pick rest of the individuals by Rank Selection, cross over and mutate.
        new_population = []
        [new_population.append(elite) for elite in elites]
        [new_population.append(self.crossover_parents()) for _ in range(len(self.individuals))]

        print("test")

        # Evolve the population while computational budget is not reached (different method -- search)

    def sort_population_fitness(self):
        """
        Sorts the population by their fitness in descending order.
        :return: Returns the sorted versions of populations and their fitnesses
        """
        list_ = list(zip(self.indv_fitness, self.individuals))  # list(), otherwise iterator ends and returns empty.
        listed = [list(a) for a in list_]

        self.indv_fitness = sorted(self.indv_fitness, reverse=True)
        self.individuals = sorted(listed, reverse=True)

    def crossover_parents(self):
        """
        Creates an action plan using crossover of two individuals from its generation.
        :return: Returns an individual for the next generation.
        """
        # Select 2 parents: (Using rank)
        select1 = random.random()
        select2 = random.random()

        diff1 = [i - select1 for i in self.cumulative_probabilities]
        parent1_idx = np.where(np.min(diff1[diff1 >= 0]))  # todo: inner part of np.min might be destructive.
        diff2 = [i - select2 for i in self.cumulative_probabilities]

        # Do crossover and form a valid sequence:

        # return new action plan that can be used to generate the new individual:
        return

    def evolve(self):
        # While computational budget is not reached - Continuously evolve the generations.
        pass

    def select_and_execute_individual(self):
        pass
