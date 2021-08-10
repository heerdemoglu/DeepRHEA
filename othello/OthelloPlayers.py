import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, player):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, player)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, player):
        # display(board)
        valid = self.game.getValidMoves(board, player)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i % self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyOthelloPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, player):
        valids = self.game.getValidMoves(board, player)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, player, a)
            score = self.game.getScore(nextBoard, player)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
