#!/bin/sh
#SBATCH --mail-user=pujanajmera1@g.ucla.edu
#SBATCH --mail-type=ALL
#SBATCH -p RM-shared 
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2

conda activate CPET_ENV_torch

compute_topo_dev_multi_process.py -o options_pycpet_generic.json
