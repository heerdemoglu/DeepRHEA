import logging

from tqdm import tqdm

from deep_rhea.RHEAPopulation import RHEAPopulation
from othello.OthelloLogic import Board

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object

        see othello/OthelloPlayers.py for an example. See pit_rhea.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        if isinstance(players[curPlayer + 1], RHEAPopulation):
            board = players[curPlayer + 1].board
        else:
            board = players[-curPlayer+1].board
        it = 0
        while self.game.getGameEnded(board.pieces, curPlayer) == 0:
            it += 1

            if isinstance(players[curPlayer + 1], RHEAPopulation):
                # action = players[curPlayer + 1].action_plan[0]
                players[curPlayer + 1].evolve()
                action = players[curPlayer + 1].select_and_execute_individual()
                players[curPlayer + 1].sort_population_fitness()
            else:
                action = players[curPlayer+1](board.pieces)

            board.pieces, curPlayer = self.game.getNextState(board.pieces, curPlayer, action)

            if isinstance(players[curPlayer+1], RHEAPopulation):
                players[curPlayer+1].set_board(board)  # if not RHEA player then set board does not exist. fix this.
            else:
                players[-curPlayer+1].set_board(board)

            if verbose:
                print('***********')
                print("Turn ", str(it), "Player ", str(-curPlayer))
                print('Action Taken:', action)
                print(board.pieces)

        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board.pieces, 1)))
            print(board.pieces)
        return curPlayer * self.game.getGameEnded(board.pieces, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        # num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

            if isinstance(self.player1, RHEAPopulation):
                self.player1.board = Board(6)
            if isinstance(self.player2, RHEAPopulation):
                self.player2.board = Board(6)

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
