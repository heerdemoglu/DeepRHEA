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
        self.args = args
        self.epsilon = 1

        self.fitness = 0  # Fitness of the individual.
        self.action_plan = action_plan  # The action plan for the individual.

        self.game = game  # Game information relayed to individual.
        self.player = player  # Player 1 is +1, player 2 is -1.
        self.board = deepcopy(board)  # Board state controlled by RHEAPopulation.
        self.nnet = nnet  # N.Network, used to fetch policy/value.

        # If individual is created from scratch, fitness is also calculated. Otherwise calculate individual's fitness.
        if self.action_plan is None:
            self.action_plan = self.build_plan()
            self.fitness *= 1/(2*self.args.INDIVIDUAL_LENGTH)
        else:
            self.measure_fitness()
            self.fitness *= 1/(2*self.args.INDIVIDUAL_LENGTH)

    def build_plan(self):
        """
            Construct's individual's action plan from scratch.
            @:return A valid action plan for this game.
        """

        temp_gamestate = deepcopy(self.game)
        temp_board = deepcopy(self.board)
        draft_plan = []

        # For each action that is to be filled in the action plan do the following:
        for i in range(self.args.INDIVIDUAL_LENGTH):

            # If game ended put -1 to the sequence: (getGameEnded outputs +1:won, -1:lose, 0:not finished, ~0 draw)
            if temp_gamestate.getGameEnded(temp_board.pieces, self.player) != 0:
                draft_plan.append(-1)
            else:
                # If game not ended: Get the best performing action from the neural network and apply it to the network:
                action, temp_gamestate, temp_board, fitness_rhea = self.plan_base_action(temp_gamestate, temp_board,
                                                                                         self.player)

                self.fitness += self.epsilon * fitness_rhea

                # Append planned action to the sequence.
                draft_plan.append(action)

                # Play the Neural Network based optimal action for the opponent as well:
                # No need to append this to the Neural network. This is to ensure validity of the action taken.
                opp_act, temp_gamestate, temp_board, fitness_opponent = self.plan_base_action(temp_gamestate,
                                                                                              temp_board, -self.player)
                # print("Picked opponent action (debug): ", opp_act)

                self.fitness += self.epsilon * fitness_opponent
                self.epsilon *= self.args.REWARD_DECAY_RATE
        return draft_plan

    def plan_base_action(self, game, board, player):
        """
        Plans 1 action that is to be taken by the agent using the neural network provided.
        :return: action, game and board states after playing a hypothetical turn.
        """
        # Plan a valid action:
        action, valid_action_indices, fitness = self.plan_valid_ply(game, board, player)

        # For each gene, there is a chance that it mutates into a random valid gene:
        if random.uniform(0, 1) >= self.args.MUTATION_CHANCE:
            #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
            np.random.shuffle(valid_action_indices)
            action = valid_action_indices[0]

        # Play the action (ply-half turn) for this player: (Used with temp board, game) - To progress construction.
        self.play_ply(game, board, player, action)

        # Append planned action to the sequence.
        return action, game, board, fitness

    def plan_valid_ply(self, game, board, player):

        temp_game = deepcopy(game)

        # Get valid indices:
        valid_action_indices = np.where(game.getValidMoves(np.array(board.pieces), player) == 1)[0]

        # If game not ended: Get the best performing action from the neural network:
        action, _ = self.nnet.predict(np.array(board.pieces) * player)  # board*player is canonical form of board.
        action = np.argmax(action)

        if action not in valid_action_indices:  # This is for safety; do not allow invalid actions in training.
            #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
            np.random.shuffle(valid_action_indices)
            action = valid_action_indices[0]

        # Hypothetically play the action and then get the fitness of the changed board state configuration:
        temp_board = temp_game.getNextState(np.array(board.pieces), player, action)[0]
        _, fitness = self.nnet.predict(np.array(temp_board) * player)  # board*player is canonical form of board.

        return action, valid_action_indices, fitness

    @staticmethod
    def play_ply(game, board, player, action):
        # Play this turn to for the player:
        board.pieces = game.getNextState(np.array(board.pieces), player, action)[0]
        move = (int(action / board.n), action % board.n)
        board.execute_move(move, player)

    def measure_fitness(self):
        """
        Measures fitness for the player and also gives best action wrt to neural network as the next move.
        :return:
        """
        temp_game = deepcopy(self.game)
        temp_board = deepcopy(self.board)
        self.fitness = 0
        self.epsilon = 1

        # Simulate the game throughout the horizon: (Opponent Modelling: RHEA)
        for i in range(len(self.action_plan)):
            action = self.action_plan[i]

            # Play this turn to for the player:
            self.play_ply(temp_game, temp_board, self.player, action)

            _, fitness_rhea = self.nnet.predict(np.array(temp_board.pieces) * self.player)
            self.fitness += self.epsilon * fitness_rhea

            # Play a best policy valid move for the opponent:
            action_opponent, valid_action_indices_opponent, fitness_opponent = \
                self.plan_valid_ply(temp_game, temp_board, -self.player)

            # Play this turn to for the opponent player:
            self.play_ply(temp_game, temp_board, -self.player, action_opponent)

        next_action, fitness_opponent = self.nnet.predict(np.array(temp_board.pieces) * self.player)
        next_action = np.argmax(next_action)

        self.fitness += self.epsilon * fitness_opponent
        self.epsilon *= self.args.REWARD_DECAY_RATE

        return next_action  # for appending the next action for the shift buffer.

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

    def append_next_action_from_nn(self):
        """Appends a valid action to the end of the current action plan."""

        next_action = self.measure_fitness()
        self.action_plan.append(next_action)
        self.fitness *= 1 / (2 * self.args.INDIVIDUAL_LENGTH)