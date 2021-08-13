import torch


from torchinfo import summary
from core_game.utils import dotdict
from othello.OthelloGame import OthelloGame
from othello.pytorch.OthelloNNet import OthelloNNet


game = OthelloGame(n=6)
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 50,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512, })

# Print network parameters:
nnet = OthelloNNet(game, args)
board_x, board_y = game.getBoardSize()
print(summary(nnet, input_size=(args.batch_size, board_x, board_y)))
