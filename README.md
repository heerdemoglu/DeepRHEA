# DeepRHEA - Deep Rolling Horizon Evolutionary Algorithm

This project aims to construct an agent that uses Deep Neural Networks and RHEA to play a game of deterministic,
competitive, zero-sum game with full information. 
Some possible games that are in this category include Chess, Go, Checkers and Othello.

# Setup and Run:

* Fork or download the repository.
* Create a new Python environment and use "requirements.yml" to set up the environment.
* pit.py is used to play an actual game between agents.
* train.py files are used to train AlphaZero and DeepRHEA agents.

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
