#!/bin/bash

# Running an array of search algorithm with random seeds 

#SBATCH --array=0-99              # there is a max array size - 512 tasks
#SBATCH -J sim 
#SBATCH --nodelist=yen10          # to be reproducible, fix the node to run simulation on
#SBATCH -p normal 
#SBATCH -n 1
#SBATCH -c 1                              # one core per task
#SBATCH -t 2-
#SBATCH --mem=40G
#SBATCH -o out/sim-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@stanford.edu

source deactivate
ml purge

ml anaconda3
source activate mathematica
python simulation-ten-seeds.py $SLURM_ARRAY_TASK_ID 
