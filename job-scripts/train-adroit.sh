#!/bin/bash
#SBATCH --job-name=iris-asterix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cb4835@princeton.edu

GAME=${GAME:-AsterixNoFrameskip-v4}
SEED=${SEED:-0}
IRIS_DIR=${IRIS_DIR:-$HOME/iris}
CONDA_ENV="iris"

source /usr/licensed/anaconda3/2024.2/etc/profile.d/conda.sh
conda activate $CONDA_ENV

export HYDRA_FULL_ERROR=1
cd $IRIS_DIR

echo "============================================"
echo "Training IRIS on: $GAME"
echo "Seed: $SEED"
echo "Dir: $IRIS_DIR"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "============================================"

python src/main.py \
    env.train.id=$GAME \
    env.test.id=$GAME \
    common.device=cuda:0 \
    common.seed=$SEED \
    wandb.mode=offline

echo "Finished at: $(date)"