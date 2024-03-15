#!/bin/sh
#SBATCH --mail-user=pujanajmera1@g.ucla.edu
#SBATCH --mail-type=ALL
#SBATCH -p RM-shared 
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=N_CORES_REQUESTED

~/programs/CPET/bin/Debug/cpet -p PQR_FILE -o options_mcpet_generic.txt -t N_CORES_REQUESTED
