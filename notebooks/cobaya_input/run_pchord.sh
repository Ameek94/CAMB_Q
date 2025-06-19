#!/bin/bash --login
###
###
#job name
#SBATCH --job-name=splinepc_u3
#job stdout file
#SBATCH --output=splinepc_u3.out
#job stderr file
#SBATCH --error=splinepc_u3.err
#maximum job time in D-HH:MM
#SBATCH --time=2-23:59
#SBATCH --account=scw2169
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#memory per process in MB 
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ameek.malhotra@swansea.ac.uk     # Where to send mail	

module load anaconda/2023.09
module load compiler/gnu/12/1.0
module load mpi/openmpi/4.1.5
module load texlive/2018 

source activate /scratch/s.ameek.malhotra/ptagw/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pmix \
     --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     cobaya-run spline_4_free_pchord.yaml --resume