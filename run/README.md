# Runtimes:

Contains all runtimes for training and agent competitions.

* best_models: best tuned networks trained for AlphaZero(MCTS)
and DeepRHEA (RHEA). Note that old models are not included with
this repository as they exceed GitHub LFS size (>10 GB). These 
models (and their respective runs) are available upon request.

* apocrita_run_othello_mcts.sh and apocrita_run_othello_rhea.sh
are used for executing code on Apocrita HPC servers provided
by QMUL.

* Confidence_value_vis.py and network_visualization.py
are helper codes that are used to create data visualizations.
(Note that the data is fed manually from game logs.)

# Main runtime scripts:
As there are two major agents with different steps required to
train and run them; there is not a single python script that runs the project code
all at once. Training can be ended anytime and the produced model can be used in a 
competition agent. Additionally, these open source codes allow playing with hyperparameters
and testing agent performances. Logs are kept automatically in described folders in the
repository.

## Competition Scripts:
* pit_mcts.py: Pits MCTS agent with other agents (MCTS as player 1.)
* pit_rhea_others.py: Pits RHEA agent with other agents (RHEA as player 1.)

## Training Scripts:
* train_othello_az.py: Trains MCTS agent with self-play,training,competition regime.
* train_othello_rhea.py: Trains RHEA agent with self-play,training,competition regime.