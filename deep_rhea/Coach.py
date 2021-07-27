import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from deep_rhea import RHEAPopulation, Arena
from othello.OthelloLogic import Board

log = logging.getLogger(__name__)


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.rhea = RHEAPopulation.RHEAPopulation(game=game, nnet=nnet, args=args, board=Board(6))

        # History of examples from args.numItersForTrainExamplesHistory latest iteration
        self.trainExamplesHistory = []

        # Can be overridden in loadTrainExamples()
        self.skipFirstSelfPlay = True

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.rhea.board
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1

            best_indv = self.rhea.evolve()
            # self.rhea.debug_print_population()
            action = self.rhea.select_and_execute_individual()
            # action = np.zeros(self.game.getActionSize())
            # action[best_indv.action_plan[0]] = 1

            trainExamples.append([board, self.curPlayer, action, None])
            board.pieces, self.curPlayer = self.game.getNextState(np.array(board.pieces), self.curPlayer, action)
            # print(' ')
            # print(board.pieces)
            # print('*******************')

            r = self.game.getGameEnded(np.array(board.pieces), self.curPlayer)  # reward is received when the game ends
            self.rhea.board = board
            # Get board, action and give the reward to the player
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
                self.rhea = RHEAPopulation.RHEAPopulation(game=self.game, nnet=self.nnet,
                                                          args=self.args, board=Board(6))

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    # In MCTS resets reach tree; RHEA resets the entire population.
                    self.rhea = RHEAPopulation.RHEAPopulation(game=self.game, nnet=self.nnet,
                                                              args=self.args, board=Board(6))
                    iterationTrainExamples += self.execute_episode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = "
                    f"{len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.save_train_examples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            board = Board(6)
            p_rhea = RHEAPopulation.RHEAPopulation(game=self.game, nnet=self.pnet, args=self.args, board=board)


            # Fixme: ValueError: setting an array element with a sequence. (After finishing self play)
            self.nnet.train(trainExamples)

            n_rhea = RHEAPopulation.RHEAPopulation(game=self.game, nnet=self.nnet, args=self.args, board=Board(6))

            # ToDo; Fix arena to play between these versions. (TypeError: 'RHEAIndividual' object is not callable at 138)
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena.Arena(p_rhea.evolve(), n_rhea.evolve(), self.game)

            p_wins, n_wins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (n_wins, p_wins, draws))
            if p_wins + n_wins == 0 or float(n_wins) / (p_wins + n_wins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    @staticmethod
    def get_checkpoint_file(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def load_train_examples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
