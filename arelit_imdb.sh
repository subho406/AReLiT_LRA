#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --account=def-amw8
#SBATCH --gpus-per-node=v100l:1         # Number of GPU(s) per node
#SBATCH --mem=30G
#SBATCH --cpus-per-task=10
#SBATCH --time=00-23:59          # time (DD-HH:MM)

# Check if seed is passed as an argument
if [ -z "$1" ]
then
  echo "No seed provided, using default seed 0"
  SEED=0
else
  SEED=$1
  echo "Using provided seed: $SEED"
fi

source ~/.bashrc
export PATH=$PATH:$CUDA_HOME

python run_train.py --dataset imdb-classification --epochs 40 --model arelit --norm none --batch_size 6 --wandb_project lra_finals_relit --jax_seed $SEED