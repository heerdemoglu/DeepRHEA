import random

import numpy as np

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
        self.pop_fitness = []
        for i in range(self.args.NUM_OF_INDIVIDUALS):
            indv = RHEAIndividual.RHEAIndividual(game=game, args=args, nnet=nnet, board=self.board)
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

    def crossover_parents(self, cum_probs):  # todo: check validity at crossover, ensure valid sequencing throughout.
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
        draft_plan = [self.individuals[parent1_idx].get_gene()[i] if i <= crossover_idx
                      else self.individuals[parent2_idx].get_gene()[i] for i in range(self.args.INDIVIDUAL_LENGTH)]

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
        for _ in range(self.args.MAX_GENERATION_BUDGET):
            # Evolve the generation for 1 step.
            self.evolve_generation()
            # self.debug_print_population()

    def select_and_execute_individual(self):
        # ToDo; Might incorporate co-evolution -- Store opponent's action plan as well and evolve both.
        #  In such case; opponent evolves the best model which is then played; also removes validity problems.
        #  However, this compresses the search space as each player individual will
        #  have only a single opponent behavior.

        # Note: (Neural Network's dual usage (for both players) mimic co-evolution.)

        # Select best individual, get its first action:
        player_action = self.individuals[0].action_plan[0]

        # Play this action in the game:
        self.board.pieces = list(
            self.game.getNextState(np.array(self.board.pieces), self.current_player, player_action)[0])
        move = (int(player_action / self.board.n), player_action % self.board.n)
        self.board.execute_move(move, self.current_player)

        # Play opponent action optimized by neural network:
        # Play a best policy valid move for the opponent:
        valid_action_indices = np.where(self.game.getValidMoves(self.board, -self.current_player) == 1)[0]
        action_opponent, _ = self.nnet.predict(np.array(self.board.pieces) * -self.current_player)

        if action_opponent in valid_action_indices:
            action_opponent = np.argmax(action_opponent)
        else:
            #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
            np.random.shuffle(valid_action_indices)
            action_opponent = valid_action_indices[0]

        print('Debug - RHEA Action Executed: ', player_action)
        print('Debug - Opponent Action Executed: ', action_opponent)

        # Play this turn to for the opponent player:
        self.board.pieces = list(
            self.game.getNextState(np.array(self.board.pieces), -self.current_player, player_action)[0])
        move = (int(action_opponent / self.board.n), action_opponent % self.board.n)
        self.board.execute_move(move, -self.current_player)

        # Pop all initial actions of all individuals, append a neural network based output at the end
        [self.individuals[i].action_plan.pop(0) for i in range(len(self.individuals))]

        # Update individual's game and boards as well.
        # Check if new board configs create validity problems in remaining; remove and replace invalid individuals.
        # ToDo: Prune invalid opponent actions and create valid sequences from the individuals that are valid.
        #  (Is it necessary?)
        for i in range(len(self.individuals)):
            self.individuals[i].game = self.game
            self.individuals[i].board = self.board

            # Append a valid (Neural network output) final action to the individual, completing the shift buffer.
            self.individuals[i].append_next_action_from_nn()

    def debug_print_population(self):
        """Code for debugging and testing purposes. Shows the individual action plans in CMD-line."""
        for indv in self.individuals:
            print('Individual plan:', indv.get_gene())
            print('Fitness:', indv.get_fitness())
            print(np.array(indv.board.pieces))
            break
        print("*******************************************************************")

    def get_indv_fitness(self):
        return self.pop_fitness
