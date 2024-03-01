#!/bin/bash
#SBATCH --time=4000


# mpirun python scripts/optuna_mpi.py -t $1 -l 1
srun python scripts/optuna_thread.py -t $1 -n $2 -l $3