import random
import RHEAIndividual

from apocrita_deeprhea.othello.OthelloLogic import Board


class RHEAPopulation:
    """
    RHEA population maintains a population of individuals and evolves them
    for generations. It also utilizes shift-buffer to carry search from past
    generations to current generations.
    """

    def __init__(self, game, nnet, args, board=None):
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
        if board is None:
            self.board = Board(6)
        else:
            self.board = board

        # Construct the list of individuals and their fitness:
        self.individuals = []
        self.indv_fitness = []
        for i in range(self.args.NUM_OF_INDIVIDUALS):
            indv = RHEAIndividual.RHEAIndividual(game=game, args=args, nnet=nnet, board=self.board)
            self.indv_fitness.append(indv.get_fitness())
            self.individuals.append(indv)

        # Sort the population with respect to their fitness: (descending order)
        self.sort_population_fitness()

    def sort_population_fitness(self):
        """
        Sorts the population by their fitness in descending order. Do not change the fitness values
        :return: Returns the sorted versions of populations and their fitness
        """
        temp_fitness = self.indv_fitness
        list_ = list(zip(temp_fitness, self.individuals))  # list(), otherwise iterator ends and returns empty.
        listed = [list(a) for a in list_]

        self.indv_fitness = sorted(self.indv_fitness, reverse=True)
        self.individuals = sorted(listed, reverse=True)

    def crossover_parents(self, cum_probs):  # fixme: validity of the plan checking!
        """
        Creates an action plan using crossover of two individuals from its generation.
        :return: Returns an individual for the next generation.
        """
        # Select 2 parents: (Using rank)
        select1 = random.random()
        select2 = random.random()

        diff1 = [i - select1 for i in cum_probs]
        parent1_idx = diff1.index(min([i for i in diff1 if i > 0]))
        diff2 = [i - select2 for i in cum_probs]
        parent2_idx = diff2.index(min([i for i in diff2 if i > 0]))

        # Do 1-point crossover and form a valid sequence: (between 0 and indv. length - 1: boundaries included)
        crossover_idx = random.randint(1, self.args.INDIVIDUAL_LENGTH - 2)  # start from first end from last index.

        # From individuals list, get the individual at index [parent_idx][1] and fetch its gene at ith index.
        draft_plan = [self.individuals[parent1_idx][1].get_gene()[i] if i <= crossover_idx
                      else self.individuals[parent2_idx][1].get_gene()[i] for i in range(self.args.INDIVIDUAL_LENGTH)]

        # return new action plan that can be used to generate the new individual:
        indv = RHEAIndividual.RHEAIndividual(game=self.game, args=self.args, nnet=self.nnet,
                                             board=self.board, action_plan=draft_plan)

        fitness = indv.get_fitness()
        fitness_indv = [fitness, indv]

        return fitness_indv

    def evolve_generation(self):
        """
        Assumes sorted individuals. Evolves the population for 1 generation.
        :return: Mutates the individuals and their fitness values after doing crossover and mutations on the current
        population.
        """
        # Pick first elite individuals (NUM_OF_BEST_INDIVIDUALS) that will ascend to the next generation.
        elites = self.individuals[:self.args.NUM_OF_BEST_INDIVIDUALS]
        elites_fitness = self.indv_fitness[:self.args.NUM_OF_BEST_INDIVIDUALS]

        # Remove elites from current generation as they will not be used in evolution.
        del self.individuals[:self.args.NUM_OF_BEST_INDIVIDUALS]
        del self.indv_fitness[:self.args.NUM_OF_BEST_INDIVIDUALS]

        # Calculate the rankings, apply them to sorted population:
        # See: https://stackoverflow.com/questions/34961489/rank-selection-in-ga
        # Normalizing fitness: fitness is based on rank ie Prob of selection.
        remaining_indv = self.args.NUM_OF_INDIVIDUALS - self.args.NUM_OF_BEST_INDIVIDUALS
        total_fitness = remaining_indv * (remaining_indv + 1) / 2

        temp_individuals = self.individuals

        for i in range(len(self.individuals)):  # each list element is a tuple with (fitness, individual)
            temp_individuals[i][0] = (self.args.NUM_OF_INDIVIDUALS -
                                      self.args.NUM_OF_BEST_INDIVIDUALS - i) / total_fitness

        # Cumulative probabilities needed to pick individuals:
        cumulative_probabilities = []
        prob = 0
        for i in range(len(self.individuals)):
            prob += temp_individuals[i][0]
            cumulative_probabilities.append(round(prob, 3))  # cap floating points at 3 decimal places.

        # Add elites to new population, pick rest of the individuals by Rank Selection, cross over and mutate.
        new_population = []
        new_fitness = []
        [new_population.append(elite) for elite in elites]
        [new_fitness.append(fitness) for fitness in elites_fitness]

        for _ in range(len(self.individuals)):
            fitness_indv = self.crossover_parents(cumulative_probabilities)
            new_population.append(fitness_indv)
            new_fitness.append(fitness_indv[0])

        # Evolve the population while computational budget is not reached (different method -- search)
        self.individuals = new_population
        self.indv_fitness = new_fitness

        # Sort the new population as well:
        self.sort_population_fitness()

    def evolve(self):
        """
        To be used by RHEA population to plan the best sequences to play.
        :return:
        """
        # Until computational budget is reached, do the following:
        for i in range(self.args.MAX_GENERATION_BUDGET):
            # Evolve the generation for 1 step.
            self.evolve_generation()
            print('gen', i)

    def select_and_execute_individual(self):
        pass

    def debug_print_population(self):
        """Code for debugging and testing purposes. Shows the individual action plans in CMD-line."""
        for indv in self.individuals:
            print('Individual plan:', indv[1].get_gene())
            print('Fitness:', indv[1].get_fitness())
