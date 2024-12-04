#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --time 6:00:00
#SBATCH --mem 16384
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

echo "job started $1"
echo "time started: $(date)"
mkdir -p $HOME/ml-project-2-teamsas/artifacts/models/$1
python main_learning.py \
    -d data/dataset.xlsx \
    -o $HOME/ml-project-2-teamsas/artifacts/models/$1 \
    -c $HOME/ml-project-2-teamsas/configs/$1.yml
echo "time finished: $(date)"
echo "job finished $1"