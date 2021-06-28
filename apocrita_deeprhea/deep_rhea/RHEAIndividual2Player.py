import copy
import math
import numpy as np
from apocrita_deeprhea.deep_rhea import Game, NeuralNet


class RHEAIndividual2Player:
    """
    Does RHE for 2 players, action plan is alternating for both players.
    """

    def __init__(self, game: Game, board, nnet: NeuralNet, args, player=1):
        """

        :param game:
        :param nnet:
        :param args:
        :param player:
        """
        self.game = game
        self.board = board
        self.nnet = nnet
        self.player = player  # Player 1 is optimized by default. Player 2 is "-1".

        self.fitness = -math.nan
        self.action_plan = []

        # Input Arguments in dotdict form.
        self.INDIVIDUAL_LENGTH = args.INDIVIDUAL_LENGTH  # How long is the action plan, for both players.
        self.MUTATION_STRENGTH = args.MUTATION_STRENGTH  # How many mutations after crossover.
        self.plan1 = args.board1  # FixMe: What to use here. This is specifically used in build_from_parents.
        self.plan2 = args.board2

        # Determine how to construct the individual.s
        if self.plan1 is None and self.plan2 is None:
            self.build_from_scratch()

        elif self.plan1 is not None and self.plan2 is not None:
            self.build_from_parents()

        else:
            raise ValueError("You need to input two parents or no parents!")

    def build_from_scratch(self):
        """
        Build individuals from scratch.
        :return: Returns the planned action set within the object.
        """

        # Do not play with the actual state of the game:
        temp_game = copy.deepcopy(self.game)
        temp_board = copy.deepcopy(self.board)

        # Until plan is filled, plan one step at a time:
        for action in range(self.INDIVIDUAL_LENGTH):
            # Play and record valid turn for first player:
            temp_board = self.simulate_turn(temp_game, self.player, temp_board)

            # Do the same for the second player: This remembers the past of the game.
            temp_board = self.simulate_turn(temp_game, -self.player, temp_board)

    def simulate_turn(self, game, player, board):
        # Find starting player's valid moves:
        valid_moves = game.getValidMoves(board, player)  # this is a numpy vector.

        # Pick a valid move play:
        valid_indices = np.where(valid_moves == 1)[0]  # gives all valid indices

        # Pick action order by random: get the first one to append to list.
        np.random.shuffle(valid_indices)
        selected_action = valid_indices[0]

        # Move the game state forward:
        move = (int(selected_action / board.n), selected_action % board.n)
        board.execute_move(move, player)

        # Save the action for this player:
        self.action_plan.append(selected_action)

        # Board config to be used later; so it is returned.
        return board

    def build_from_parents(self):
        # Do not play with the actual state of the game:
        temp_game = copy.deepcopy(self.game)
        temp_board = copy.deepcopy(self.board)

        # Build boolean sequence: 0 for parent 1, 1 for parent 2.
        crossover_idx = np.random.randint(2, size=2*self.INDIVIDUAL_LENGTH)

        # Build the sequence: (Does uniform crossover)
        # If crossover index is zero; take the value from parent 1, else from parent 2 for all indices available.
        draft_plan = [self.plan1.action_plan[i] if crossover_idx[i] == 0
                      else self.plan2.action_plan[i] for i in range(self.INDIVIDUAL_LENGTH)]

        # Repair invalid actions: Mutate as less actions as possible while generating the new sequence:
        # for action in self.action_plan:
        for i in range(len(draft_plan)):

            # Mask player turns: +1 for player 1, -1 for player 2.
            p = [self.player if i % 2 == 0 else -self.player for i in range(len(draft_plan))]

            # Repair the genome:
            draft_plan, temp_game, temp_board = repair_genome(self.draft_plan[i], draft_plan, temp_game, p[i], temp_board)

        # Implement the new plan:
        self.action_plan = draft_plan

    # FixMe: action not in valid indices; the namings are incorrect.
    def repair_genome(self, action, plan, game, player, board):
        # See the list of valid locations available:
        valid_moves = game.getValidMoves(board, player)
        valid_indices = np.where(valid_moves == 1)[0]  # gives all valid move locations.

        # If current action is not in this set, mutate to a random valid move, else continue:
        if action not in valid_indices:
            np.random.shuffle(valid_indices)
            selected_action = valid_indices[0]

            # Move the game state forward:
            move = (int(selected_action / board.n), selected_action % board.n)
            board.execute_move(move, self.player)

            # Mutate the corresponding entry:
            plan[np.where(action)[0][0]] = selected_action

            # Return plan as it will be used later.
            return plan, game, board

    def calculate_fitness(self):
        # ToDo: Write this part.
        pass
