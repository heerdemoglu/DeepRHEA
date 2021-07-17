import random
import numpy as np

from copy import deepcopy
from apocrita_deeprhea.deep_rhea.Game import Game


class RHEAIndividual:
    """
    Each RHEA Individual handles their operations themselves and reports
    to RHEAPopulation.
    """

    def __init__(self, game: Game, args, nnet, board=None, action_plan=None, player=1):
        self.INDIVIDUAL_LENGTH = args.INDIVIDUAL_LENGTH  # How long is the action plan.
        self.MUTATION_CHANCE = args.MUTATION_CHANCE  # Chance of having a mutation on a gene.
        self.action_plan = action_plan  # The action plan for the individual.
        self.fitness = None  # Fitness of the individual.
        self.game = game  # Game information relayed to individual.
        self.player = player  # Player 1 is +1, player 2 is -1.
        self.board = deepcopy(board)  # Board state controlled by RHEAPopulation.
        self.nnet = nnet  # N.Network, used to fetch policy/value.

        # If individual is created from scratch, fitness is also calculated. Otherwise calculate individual's fitness.
        if self.action_plan is None:
            self.action_plan, self.fitness = self.build_plan()
        else:
            self.fitness = self.measure_fitness()

    def build_plan(self):
        """
            Construct's individual's action plan from scratch.
            @:return A valid action plan for this game.
        """

        temp_gamestate = deepcopy(self.game)
        temp_board = deepcopy(self.board)
        draft_plan = []

        # For each action that is to be filled in the action plan do the following:
        for i in range(self.INDIVIDUAL_LENGTH):

            # If game ended put -1 to the sequence: (getGameEnded outputs +1:won, -1:lose, 0:not finished, ~0 draw)
            if temp_gamestate.getGameEnded(temp_board.pieces, self.player) != 0:
                draft_plan.append(-1)
            else:
                # If game not ended: Get the best performing action from the neural network and apply it to the network:
                action, temp_gamestate, temp_board = self.plan_base_action(temp_gamestate, temp_board,
                                                                           self.player)

                # Append planned action to the sequence.
                draft_plan.append(action)

                # Play the Neural Network based optimal action for the opponent as well:
                # No need to append this to the Neural network. This is to ensure validity of the action taken.
                opp_act, temp_gamestate, temp_board = self.plan_base_action(temp_gamestate, temp_board,
                                                                            -self.player)
                # print("Picked opponent action (debug): ", opp_act)

        # Final board configuration also estimates the fitness of the individual.
        _, fitness = self.nnet.predict(self.player * (np.array(temp_board.pieces)))

        return draft_plan, fitness

    def plan_base_action(self, game, board, player):
        """
        Plans 1 action that is to be taken by the agent using the neural network provided.
        :return: action, game and board states after playing a hypothetical turn.
        """
        # Get valid indices:
        valid_action_indices = np.where(game.getValidMoves(board, player) == 1)[0]

        # If game not ended: Get the best performing action from the neural network:
        action, _ = self.nnet.predict(np.array(board.pieces) * player)  # board*player is canonical form of board.

        if action in valid_action_indices:
            action = np.argmax(action)
        else:
            #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
            np.random.shuffle(valid_action_indices)
            action = valid_action_indices[0]

        # For each gene, there is a chance that it mutates into a random valid gene:
        if random.uniform(0, 1) >= self.MUTATION_CHANCE:
            #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
            np.random.shuffle(valid_action_indices)
            action = valid_action_indices[0]

        # Play this turn to for the player:
        board.pieces = game.getNextState(np.array(board.pieces), player, action)[0]
        move = (int(action / board.n), action % board.n)
        board.execute_move(move, player)

        # Append planned action to the sequence.
        return action, game, board

    def measure_fitness(self):
        temp_game = deepcopy(self.game)
        temp_board = deepcopy(self.board)

        # Simulate the game throughout the horizon: (Opponent Modelling: RHEA)
        for i in range(len(self.action_plan)):
            action = self.action_plan[i]

            # Play this turn to for the player:
            temp_board.pieces = temp_game.getNextState(np.array(temp_board.pieces), self.player, action)[0]
            move = (int(action / temp_board.n), action % temp_board.n)
            temp_board.execute_move(move, self.player)

            # Play a best policy valid move for the opponent:
            valid_action_indices = np.where(temp_game.getValidMoves(temp_board, -self.player) == 1)[0]
            action_opponent, _ = self.nnet.predict(np.array(temp_board.pieces) * -self.player)

            if action_opponent in valid_action_indices:
                action_opponent = np.argmax(action_opponent)  # fixme: bug? action is player's action, not opponents.
            else:
                #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
                np.random.shuffle(valid_action_indices)
                action_opponent = valid_action_indices[0]

            # Play this turn to for the opponent player:
            temp_board.pieces = temp_game.getNextState(np.array(temp_board.pieces), -self.player, action_opponent)[0]
            move = (int(action_opponent / temp_board.n), action_opponent % temp_board.n)
            temp_board.execute_move(move, -self.player)

        # If game not ended: Get the best performing action from the neural network:
        # temp_board*cls.player is canonical form of board.
        _, fitness = self.nnet.predict(np.array(temp_board.pieces) * self.player)

        return fitness

    def get_gene(self):
        """
        :return: Returns the action plan of the individual.
        """
        return self.action_plan

    def set_gene(self, action_plan):
        """

        :param action_plan:
        :return:
        """
        self.action_plan = action_plan

    def get_fitness(self):
        return self.fitness
