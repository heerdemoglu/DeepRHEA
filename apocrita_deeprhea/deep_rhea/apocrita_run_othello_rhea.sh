#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -M ec20041@qmul.ac.uk # Sends notifications email to this address
#$ -m bea # Emails are sent on begin, end and abortion
#$ -N rhea_othello_full
#$ -o rhea_othello_full
#$ -m beas

module load anaconda3
conda activate thesis

cd /data/scratch/$USER/apocrita_deeprhea   # Move to the scratch directory


# Replace the following line with a program or command
python train_othello_rhea.py 'full_training'
