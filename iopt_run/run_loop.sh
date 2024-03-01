#!/bin/bash
#SBATCH --time=4000

for n in 1 4 8 16
do
    srun python scripts/iopt.py -t $1 -n $n -l 10 -a $2 -d $3
done 

