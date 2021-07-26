import random

import numpy as np

import deep_rhea.RHEAIndividual as RHEAIndividual

from othello.OthelloLogic import Board


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
        self.player = 1  # Always start with player 1. (design choice)

        # Get the initial board configuration to set up the individuals.
        if board is None:
            self.board = Board(6)
        else:
            self.board = board

        # Construct the list of individuals and their fitness:
        self.individuals = []
        self.pop_fitness = []
        for i in range(self.args.NUM_OF_INDIVIDUALS):
            indv = RHEAIndividual.RHEAIndividual(game=game, args=args, nnet=nnet, board=board)
            self.pop_fitness.append(indv.get_fitness())
            self.individuals.append(indv)

        # Sort the population with respect to their fitness: (descending order)
        self.sort_population_fitness()

    def sort_population_fitness(self):
        """
        Sorts the population by their fitness in descending order. Do not change the fitness values
        :return: Returns the sorted versions of populations and their fitness
        """
        temp_fitness = self.pop_fitness
        list_ = list(zip(temp_fitness, self.individuals))  # list(), otherwise iterator ends and returns empty.
        listed = [list(a) for a in list_]

        self.pop_fitness = sorted(self.pop_fitness, reverse=True)
        temp_indvs = sorted(listed, reverse=True, key=lambda x: x[0])

        for i in range(len(self.individuals)):
            self.individuals[i] = temp_indvs[i][1]

    def crossover_parents(self, cum_probs):
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

        parent1 = self.individuals[parent1_idx]
        parent2 = self.individuals[parent2_idx]

        # Decide on crossover index:
        crossover_idx = random.randint(1, self.args.INDIVIDUAL_LENGTH - 1)  # start from first end from last index

        # From individuals list, get the individual at index [parent_idx][1] and fetch its gene at ith index.
        draft_plan = [parent1.get_gene()[i] if i <= crossover_idx
                      else parent2.get_gene()[i] for i in range(self.args.INDIVIDUAL_LENGTH)]

        draft_opp_plan = [parent1.get_opponent_gene()[i] if i <= crossover_idx
                          else parent2.get_opponent_gene()[i]
                          for i in range(self.args.INDIVIDUAL_LENGTH)]

        # Create and Return child individual along with its fitness:
        indv = RHEAIndividual.RHEAIndividual(game=self.game, args=self.args, nnet=self.nnet,
                                             board=self.board, action_plan=draft_plan, opp_plan=draft_opp_plan)

        # Mutate and repair the genes of the new individual.
        for i in range(self.args.CROSSOVER_MUTATIONS):
            idx = random.randint(1, self.args.INDIVIDUAL_LENGTH - 1)
            indv.mutate_genes(idx)

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
        elites_fitness = self.pop_fitness[:self.args.NUM_OF_BEST_INDIVIDUALS]

        # Remove elites from current generation as they will not be used in evolution.
        del self.individuals[:self.args.NUM_OF_BEST_INDIVIDUALS]
        del self.pop_fitness[:self.args.NUM_OF_BEST_INDIVIDUALS]

        # Calculate the rankings, apply them to sorted population:
        # See: https://stackoverflow.com/questions/34961489/rank-selection-in-ga
        # Normalizing fitness: fitness is based on rank ie Prob of selection.
        remaining_indv = self.args.NUM_OF_INDIVIDUALS - self.args.NUM_OF_BEST_INDIVIDUALS
        total_fitness = remaining_indv * (remaining_indv + 1) / 2

        temp_individuals_fitness = self.pop_fitness

        for i in range(len(self.pop_fitness)):  # each list element is a tuple with (fitness, individual)
            temp_individuals_fitness[i] = (self.args.NUM_OF_INDIVIDUALS -
                                           self.args.NUM_OF_BEST_INDIVIDUALS - i) / total_fitness

        # Cumulative probabilities needed to pick individuals:
        cumulative_probabilities = []
        prob = 0
        for i in range(len(self.individuals)):
            prob += temp_individuals_fitness[i]
            cumulative_probabilities.append(round(prob, 3))  # cap floating points at 3 decimal places.

        # Add elites to new population, pick rest of the individuals by Rank Selection, cross over and mutate.
        new_population = []
        new_fitness = []
        [new_population.append(elite) for elite in elites]
        [new_fitness.append(fitness) for fitness in elites_fitness]

        for _ in range(len(self.individuals)):
            fitness_indv = self.crossover_parents(cumulative_probabilities)
            new_population.append(fitness_indv[1])
            new_fitness.append(fitness_indv[0])

        # Evolve the population while computational budget is not reached (different method -- search)
        self.individuals = new_population
        self.pop_fitness = new_fitness

        # Sort the new population as well:
        self.sort_population_fitness()

    def evolve(self):
        """
        To be used by RHEA population to plan the best sequences to play.
        :return:
        """
        # Until computational budget is reached, do the following:
        for j in range(self.args.MAX_GENERATION_BUDGET):
            # if (j+1) % 10 == 0:
            #     print('Generation ', j + 1, ' computed.')
            # for i in range(len(self.individuals)):
            #     print('Individual ', i + 1, '  -  ', self.individuals[i].get_gene(), 'Fitness: ',
            #           self.individuals[i].get_fitness())
            # print('')
            # Evolve the generation for 1 step.
            self.evolve_generation()
            # self.debug_print_population()
            self.sort_population_fitness()
        # action = self.select_and_execute_individual()
        return self.individuals[0]

    def select_and_execute_individual(self):
        # This is used for testing the population evolution process on RHEAPopulationTest.py.
        # Might incorporate co-evolution -- Store opponent's action plan as well and evolve both.
        #  In such case; opponent evolves the best model which is then played; also removes validity problems.
        #  However, this compresses the search space as each player individual will
        #  have only a single opponent behavior.

        # Note: (Neural Network's dual usage (for both players) mimic co-evolution.)

        # Select best individual, get its first action:
        player_action = self.individuals[0].action_plan[0]

        # Play this action in the game:
        self.individuals[0].play_ply(self.game, self.board, self.player, player_action)

        # Play opponent action optimized by neural network:
        # Play a best policy valid move for the opponent:
        action_opponent, valid_action_indices, fitness = \
            self.individuals[0].plan_valid_ply(self.game, self.board, -self.player)

        # print('Debug - Individual Plan: ', self.individuals[0].get_gene())
        # print('Debug - RHEA (+1) Action Executed: ', player_action)
        # print('Debug - Opponent (-1) Action Executed: ', action_opponent)

        # Play this turn to for the opponent player:
        self.individuals[0].play_ply(self.game, self.board, -self.player, action_opponent)

        # Pop all initial actions of all individuals, append a neural network based output at the end
        [self.individuals[i].action_plan.pop(0) for i in range(len(self.individuals))]
        [self.individuals[i].opp_plan.pop(0) for i in range(len(self.individuals))]

        # Update individual's game and boards as well.
        # Check if new board configs create validity problems in remaining; remove and replace invalid individuals.
        for i in range(len(self.individuals)):
            self.individuals[i].game = self.game
            self.individuals[i].board = self.board

            # Append a valid (Neural network output) final action to the individual, completing the shift buffer.
            _, _ = self.individuals[i].append_next_action_from_nn()  # next action for players are already handled.

        # print(self.debug_print_population())
        return player_action

    def debug_print_population(self):
        """Code for debugging and testing purposes. Shows the individual action plans in CMD-line."""
        print('Remaining Individual plan:', self.individuals[0].get_gene())
        # for i in range(len(self.individuals)):
        #     print('Individual ', i+1)
        #     print(self.individuals[i].get_gene())
        print('Fitness:', self.individuals[0].get_fitness())
        print(np.array(self.individuals[0].board.pieces))
        print('Score for RHEA Agent: ', self.game.getScore(self.individuals[0].board.pieces, self.player))
        print("*******************************************************************")

    def get_indv_fitness(self):
        return self.pop_fitness
