#!/bin/bash

# Running an array of search algorithm with random seeds 

#SBATCH --array=0-499             # there is a max array size - 500 tasks
#SBATCH -J mu3_sim 
#SBATCH --nodelist=yen17          # to be reproducible, fix the node to run simulation on
#SBATCH -p normal 
#SBATCH -n 1
#SBATCH -c 1                              # one core per task
#SBATCH -t 1-
#SBATCH -o out/sim-%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user@stanford.edu

source deactivate
ml purge

source /zfs/projects/faculty/sjc-recombinant-search/nrapstin/new-preference/venv/bin/activate

for i in {0..19}; do
    python simulation-one-seed.py $((SLURM_ARRAY_TASK_ID + i * 500))
done
