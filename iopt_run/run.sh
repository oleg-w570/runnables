#!/bin/bash
#SBATCH --time=4000

srun python scripts/iopt.py -t $1 -n $2 -l 10 -a $3 -d $4 