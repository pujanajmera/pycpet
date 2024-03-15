#!/bin/bash
#SBATCH --job-name=distcomp
#SBATCH -p RM-shared
#SBATCH --time=5:00:00
#SBATCH --mail-user=ajmera
#SBATCH -A che230029p
#SBATCH --mail-type=ALL


OPTIONS=./options_comp.json

dist_and_compress_each_protein.py --options $OPTIONS
