from copy import copy

import numpy as np

from apocrita_deeprhea.deep_rhea.Game import Game


class RHEAIndividual:
    """
    Each RHEA Individual handles their operations themselves and reports
    to RHEAPopulation.
    """

    def __init__(self, game: Game, args, board=None, player=1, parent1=None, parent2=None):
        self.INDIVIDUAL_LENGTH = args.INDIVIDUAL_LENGTH  # How long is the action plan.
        self.MUTATION_STRENGTH = args.MUTATION_STRENGTH  # How many mutations after crossover.
        self.parent1 = parent1  # If exists, the first associated parent.
        self.parent2 = parent2  # If exists, the second parent.
        self.action_plan = None  # The action plan for the individual.
        self.fitness = None  # Fitness of the individual.
        self.game = game  # Game information relayed to individual.
        self.player = player  # Player 1 is +1, player 2 is -1

        # Set the board of the game, this has to be changed every generation.
        if board is None:
            self.board = self.game.getInitBoard()
        else:
            self.board = board

        # Set the action plan:
        if parent1 is not None and parent2 is not None:
            self.build_from_parents()

        # If there are no parents: Construct a random sequence.
        elif parent2 is None and parent2 is None:
            self.build_from_scratch(self.board)

        else:
            raise ValueError("You need to input two parents or no parents!")

        # Execute the action plan and learn the fitness:
        self.measure_fitness(board)

    def build_from_parents(self):
        """
        From the current state of the game, build a VALID action plan of given
        individual length and parents. Mutations must return a valid action plan.

        :return: Returns a valid action plan with uniform crossover and random mutation.
        """

        # Build boolean sequence: 0 for parent 1, 1 for parent 2.
        crossover_idx = np.random.randint(2, size=self.INDIVIDUAL_LENGTH)
        print('Crossover idx:', crossover_idx)
        # Build the sequence: (Does uniform crossover)
        # If crossover index is zero; take the value from parent 1, else from parent 2 for all indices available.
        draft_plan = [self.parent1.action_plan[i] if crossover_idx[i] == 0
                      else self.parent2.action_plan[i] for i in range(self.INDIVIDUAL_LENGTH)]

        # ToDo: Assure validity after cross-over.

        self.action_plan = draft_plan

    # ToDo: Valid move checking. This should be progressing over time. P1 then P2 then P1 and so on.
    def build_from_scratch(self, board):
        """
        Get valid moves from the game and build a sequence from scratch. Sets the action plan. Returns -1 for genes
        if there are enough valid moves available given individual length.
        :param board: Board positioning is provided by the Coach
        """

        temp_gamestate = copy(self.game)
        temp_board = copy(board)
        draft_plan_indices = []

        # Progressively construct the gene:
        for i in range(self.INDIVIDUAL_LENGTH):

            # Find valid moves:
            valid_moves = temp_gamestate.getValidMoves(temp_board, self.player)  # this is a numpy vector.

            # Pick a valid move play:
            valid_indices = np.where(valid_moves == 1)[0]  # gives all valid indices

            # Pick action order by random: get the first one to append to list.
            np.random.shuffle(valid_indices)
            selected_action = valid_indices[0]

            # Play the action on temporary board, append selected action to gene.
            # temp_board.pieces = temp_gamestate.getNextState(temp_board, self.player, selected_action)
            move = (int(selected_action / temp_board.n), selected_action % temp_board.n)
            temp_board.execute_move(move, self.player)

            draft_plan_indices = np.append(draft_plan_indices, selected_action)

            # Play the turn for the competitor; random action selection:
            competitor_valid_moves = self.game.getValidMoves(temp_board, -self.player)
            competitor_valid_indices = np.where(competitor_valid_moves == 1)[0]  # gives all valid indices
            np.random.shuffle(competitor_valid_indices)
            competitor_selected_action = competitor_valid_indices[0]
            # temp_board.getNextState(temp_board, -self.player, competitor_selected_action)
            competitor_move = (int(competitor_selected_action / temp_board.n), competitor_selected_action % temp_board.n)
            temp_board.execute_move(competitor_move, -self.player)

        self.action_plan = draft_plan_indices

    # ToDo: Complete this:
    def measure_fitness(self, board):
        pass

    def set_action_plan(self, gene):
        """
        Sets the action plan to the gene provided.
        :param gene: Gene sequence replace object's action plan.
        :return: Replaces current action plan with the plan specified.
        """
        self.action_plan = gene

    def get_gene(self):
        """
        :return: Returns the action plan of the individual.
        """
        return self.action_plan

    def set_board_status(self, board):
        """
        Sets the board status known by the individual to  new board status
        :param board:
        :return:
        """
        self.board = board
