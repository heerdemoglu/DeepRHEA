import logging
import math
import numpy as np

EPS = 1e-8
log = logging.getLogger(__name__)


class RHEA:

    def __init__(self, game, nnet, args):
        raise NotImplementedError

    def getActionProb(self, canonicalBoard, temp=1):
        raise NotImplementedError

    def search(self, canonicalBoard):
        raise NotImplementedError