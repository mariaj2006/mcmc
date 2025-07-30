#!/bin/bash

#SBATCH --job-name="velocity_fitting"
#SBATCH --time=10:00:00
#SBATCH --array=1-2
#SBATCH --mem=2G
#SBATCH --output=out-array_%A_%a.out  
#SBATCH --error=err-array_%A_%a.err  
#SBATCH --partition=obs
#SBATCH --ntasks=1


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load python
conda activate emcee_fitting

python3 mcmc_fitting/scripts/O3_1comp.py $SLURM_ARRAY_TASK_ID









