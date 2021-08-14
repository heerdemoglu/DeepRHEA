import random
import numpy as np

from copy import deepcopy
from core_game.Game import Game


#  Same action plans may lead to different board configs due to opponent behavior. Different boards correspond to
#  different fitness; which is not distinguishable by the individual only by using action plan.
#  Maybe it is best to double the plan size; including opponent. Play two ply and re-order individuals so that opponent
#  models are kept valid.
from othello.OthelloLogic import Board


class RHEAIndividual:
    """
    Each RHEA Individual handles their operations themselves and reports
    to RHEAPopulation.
    """

    def __init__(self, game: Game, args, nnet, board=None, action_plan=None, opp_plan=None, player=1):
        self.args = args

        self.fitness = 0                # Fitness of the individual.
        self.action_plan = action_plan  # The action plan for the individual.
        self.opp_plan = opp_plan   # Hypothetical opponent plan for this individual.

        self.game = game                # Game information relayed to individual.
        self.player = player            # Player 1 is +1, player 2 is -1.
        self.board = deepcopy(board)    # Board state controlled by RHEAPopulation.
        self.nnet = nnet      # Neural Network, used to fetch policy/value.

        # If individual is created from scratch, fitness is also calculated. Otherwise calculate individual's fitness.
        if self.action_plan is None or self.opp_plan is None:
            self.action_plan, self.opp_plan, self.fitness = self.build_plan()
        else:
            _ = self.measure_fitness()  # next action is not useful here; learning the fitness is sufficient.

    def build_plan(self):
        """
            Construct's individual's action plan from scratch. Plays the plan (and opponent plan) on a copy of the
            board configuration to build the action plan for the game.
            @:return A valid action plan for this game.
        """

        temp_gamestate = deepcopy(self.game)
        temp_board = deepcopy(self.board)
        fitness = self.fitness
        draft_plan = []
        opp_plan = []

        # For each action that is to be filled in the action plan do the following:
        for i in range(self.args.INDIVIDUAL_LENGTH):

            # If game ended put -1 to the sequence: (getGameEnded outputs +1:won, -1:lose, 0:not finished)
            if temp_gamestate.getGameEnded(temp_board.pieces, self.player) != 0:
                draft_plan.append(36)
                opp_plan.append(36)
                _, fitness = self.nnet.predict(np.array(temp_board.pieces) * self.player)
            else:
                # If game not ended: Get the best performing action from the neural network and apply it to the network:
                action, temp_gamestate, temp_board, _ = self.plan_base_action(temp_gamestate, temp_board, self.player)

                # Append planned action to the sequence.
                draft_plan.append(action)

                # Play the Neural Network based optimal action for the opponent as well:
                # No need to append this to the Neural network. This is to ensure validity of the action taken.
                # Even no-op (36) is played; it is still a move. Opponent does not explore in this case (mutate=False).
                opp_act, temp_gamestate, temp_board, _ = self.plan_base_action(temp_gamestate, temp_board,
                                                                               -self.player, mutate=False)
                opp_plan.append(opp_act)

            # Determine the fitness for the player for this board configuration.
            _, fitness = self.nnet.predict(np.array(temp_board.pieces) * self.player)

        # Final move played by the opponent; which gives the fitness of the board state (current state) after opponent.
        return draft_plan, opp_plan, fitness

    def plan_base_action(self, game, board, player, mutate=True):
        """
        Plans 1 action that is to be taken by the agent using the neural network provided.
        :return: action, game and board states after playing a hypothetical turn.
        """
        # Plan a valid action:
        action, valid_action_indices, fitness = self.plan_valid_ply(game, board, player)

        # For each gene, there is a chance that it mutates into a random valid gene:
        if mutate:
            if random.uniform(0, 1) >= self.args.MUTATION_CHANCE:
                #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
                np.random.shuffle(valid_action_indices)
                action = valid_action_indices[0]

        # Play the action (ply-half turn) for this player: (Used with temp board, game) - To progress construction.
        self.play_ply(game, board, player, action)

        # Append planned action to the sequence.
        return action, game, board, fitness

    def plan_valid_ply(self, game, board, player):
        """
        Plans a valid half-turn given the board config player id and game rules.
        """

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
        """
        Executes the action in given game board.
        """
        # Play this turn to for the player:
        board.pieces = game.getNextState(np.array(board.pieces), player, action)[0]
        move = (int(action / board.n), action % board.n)
        board.execute_move(move, player)

    def measure_fitness(self):
        """
        Measures fitness for the player and also plans for the next states.
        :return:
        """
        temp_game = deepcopy(self.game)
        temp_board = deepcopy(self.board)

        # Simulate the game throughout the horizon:
        for i in range(len(self.action_plan)):
            action = self.action_plan[i]
            opp_action = self.opp_plan[i]

            # Play this ply to for the player:
            self.play_ply(temp_game, temp_board, self.player, action)

            # Play ply of opponent:
            self.play_ply(temp_game, temp_board, -self.player, opp_action)

        # From Neural Network get a new action fo the shift buffer (more trained --> less random)
        next_action, _, _, _ = self.plan_base_action(temp_game, temp_board, self.player)
        self.play_ply(temp_game, temp_board, self.player, next_action)

        # Get a new opponent action
        next_opponent_action, _, _, _ = self.plan_base_action(temp_game, temp_board, -self.player)
        self.play_ply(temp_game, temp_board, -self.player, next_opponent_action)

        _, self.fitness = self.nnet.predict(np.array(temp_board.pieces) * self.player)

        return next_action, next_opponent_action  # for appending the next action for the shift buffer.

    def append_next_action_from_nn(self):
        """Appends a valid action to the end of the current action plan."""

        next_action, next_opp_action = self.measure_fitness()
        self.action_plan.append(next_action)
        self.opp_plan.append(next_opp_action)
        return next_action, next_opp_action

    def get_gene(self):
        """
        :return: Returns the action plan of the individual.
        """
        return self.action_plan

    def get_opponent_gene(self):
        return self.opp_plan

    def set_gene(self, action_plan):
        """

        :param action_plan:
        :return:
        """
        self.action_plan = action_plan

    def get_fitness(self):
        return self.fitness

    def mutate_genes(self, index):
        """
        Mutate the gene at the given index and adjust the remaining sequence as close to the original sequence.
        Mutations return valid action sets, given the current board configuration.
        """
        temp_board = Board(6)

        # Play the hypothetical game until index is reached:
        for i in range(index):
            self.play_ply(self.game, temp_board, self.player, self.action_plan[i])
            self.play_ply(self.game, temp_board, -self.player, self.opp_plan[i])

        # Mutate the action plan for the RHEA player with a random valid action:
        valid_action_indices = np.where(self.game.getValidMoves(np.array(temp_board.pieces), self.player) == 1)[0]
        np.random.shuffle(valid_action_indices)
        action = valid_action_indices[0]

        # Reflect the changes and play this:
        self.action_plan[index] = action
        self.play_ply(self.game, temp_board, self.player, action)

        # Check opponent action validity mutate it as well if it becomes invalid:
        valid_action_indices = np.where(self.game.getValidMoves(np.array(temp_board.pieces), -self.player) == 1)[0]
        if self.opp_plan[index] not in valid_action_indices:
            #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
            np.random.shuffle(valid_action_indices)
            opp_action = valid_action_indices[0]
            self.opp_plan[index] = opp_action

        self.play_ply(self.game, temp_board, -self.player, self.opp_plan[index])

        # Repair procedure: (Check the rest)
        for j in range(index+1, len(self.action_plan)):
            # First RHEA Player:
            self.action_plan[j] = action
            valid_actions = np.where(self.game.getValidMoves(np.array(temp_board.pieces), self.player) == 1)[0]

            if action not in valid_actions:
                #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
                np.random.shuffle(valid_actions)
                self.action_plan[j] = valid_actions[0]

            self.play_ply(self.game, temp_board, self.player, action)

            # Now, opponent player:
            self.opp_plan[j] = action
            valid_actions = np.where(self.game.getValidMoves(np.array(temp_board.pieces), -self.player) == 1)[0]

            if action not in valid_actions:
                #  Returns all possible valid indices, then randomize and get the first action on list of valid actions.
                np.random.shuffle(valid_actions)
                self.opp_plan[j] = valid_actions[0]

            self.play_ply(self.game, temp_board, -self.player, action)
