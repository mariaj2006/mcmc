#!/bin/bash

#SBATCH --job-name="convergence test"
#SBATCH --time=10:00:00
##SBATCH --array=1-100
#SBATCH --mem=3G
#SBATCH --output=/carnegie/nobackup/scratch/msanchezrincon/slurm_output/chain/vel_out-array_%A_%a.out  
#SBATCH --error=/carnegie/nobackup/scratch/msanchezrincon/slurm_output/chain/vel_err-array_%A_%a.err  
#SBATCH --partition=obs
#SBATCH --ntasks=1


##echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load python
conda activate emcee_fitting

python3 mcmc/scripts/vmap.py ##$SLURM_ARRAY_TASK_ID
