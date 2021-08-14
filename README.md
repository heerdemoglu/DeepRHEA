# DeepRHEA - Deep Rolling Horizon Evolutionary Algorithm

This project aims to construct an agent that uses Deep Neural Networks and RHEA to play a game of deterministic,
competitive, zero-sum game with full information. 
Some possible games that are in this category include Chess, Go, Checkers and Othello.

# Setup and Run:

* Fork or download the repository.
* Create a new Python environment and use "requirements.yml" to set up the environment.
* pit.py is used to play an actual game between agents.
* train.py files are used to train AlphaZero and DeepRHEA agents.

Note: Training examples, logs, models and runtimes are not included with this GitHub repository as it exceeds LFS sizes. These files are available upon request.

## Main runtime scripts:
As there are two major agents with different steps required to
train and run them; there is not a single python script that runs the project code
all at once. Training can be ended anytime and the produced model can be used in a 
competition agent. Additionally, these open source codes allow playing with hyperparameters
and testing agent performances. Logs are kept automatically in described folders in the
repository.

### Competition Scripts:
* pit_mcts.py: Pits MCTS agent with other agents (MCTS as player 1.)
* pit_rhea_others.py: Pits RHEA agent with other agents (RHEA as player 1.)

### Training Scripts:
* train_othello_az.py: Trains MCTS agent with self-play,training,competition regime.
* train_othello_rhea.py: Trains RHEA agent with self-play,training,competition regime.

# Referenced Repositories:

The aim of this project is to make an analogous version to AlphaGo using RHEA instead of MCTS. 
This project extends upon the following repository. RHEAIndividual and RHEAPopulation is written to replace
MCTS with RHEA implementation. Some modifications to the Arena.py, Coach.py and Pit.py was made to accommodate

## Alpha Zero General, Surag Nair et. al. 

The repository is created by Surag Nair and the repository can be reached through [this](https://github.com/suragnair/alpha-zero-general) link.

This repository implements some games where agents inspired from AlphaGo is trained. This repository is used as a baseline to train AlphaGo agent; where DeepRHEA agent 
will be coupled with the same environment for comparison purposes.

# Package Structure:

* alpha_zero: Contains AlphaZero related code and logic used by Nair et al.
* core_game: Has interfaces for implementing games other than Othello; any new game should extend these
templates.
* deep_rhea: Contains modified game control scripts and logic for RHEA.
* othello: Contains game logic libraries and neural network descriptions for Othello.
* run: Contains code for training and testing scripts as well as trained models.
  (Training examples and checkpoints are excluded in this repository, only best models are uploaded for 
   reference)


# Acknowledgements:
* I would like to thank my project supervisor, Dr. Diego Perez-Liebana for his support and guidance throughout this project. 
* This research utilised Queen Mary's Apocrita HPC facility, supported by QMUL Research-IT. http://doi.org/10.5281/zenodo.438045
