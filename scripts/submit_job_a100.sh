#!/bin/bash
#SBATCH -o /home/minghanwu/jobs/%j-%x.stdout
#SBATCH -e /home/minghanwu/jobs/%j-%x.stderr
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=15G
#SBATCH --partition=A100
#SBATCH --gres=gpu:2
# SBATCH --exclude=node[01-17]
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minghan.wang@monash.edu
echo $@
bash $@