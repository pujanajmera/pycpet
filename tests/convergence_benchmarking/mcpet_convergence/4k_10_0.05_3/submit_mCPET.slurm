#!/bin/sh
#SBATCH --mail-user=pujanajmera1@g.ucla.edu
#SBATCH --mail-type=ALL
#SBATCH -p RM-shared 
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32

cpet -p 4k_charges.pqr -o options_mcpet_generic.txt -t 32
