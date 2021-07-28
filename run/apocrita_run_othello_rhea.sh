#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -M ec20041@qmul.ac.uk # Sends notifications email to this address
#$ -m bea # Emails are sent on begin, end and abortion
#$ -N deep_rhea_full_training
#$ -o deep_rhea_full_training
#$ -m beas

module load anaconda3

conda create --quiet --yes --name thesis
conda activate thesis

cd /data/scratch/$USER/DeepRHEA/run   # Move to the scratch directory


# Replace the following line with a program or command
python train_othello_rhea.py 'deep_rhea_full_training'
