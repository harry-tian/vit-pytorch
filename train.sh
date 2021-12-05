#!/bin/bash
#
#SBATCH --mail-user=tianh@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/tianh/slurm/out/%j.%N.stdout
#SBATCH --error=/home/tianh/slurm/stderr/%j.%N.stderr
#SBATCH --job-name=vit
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
# #SBATCH --chdir=/net/scratch/hanliu/radiology/explain_teach/model

hostname
echo $CUDA_VISIBLE_DEVICES

python train.py \
  --patch=8 \
  --wandb_run_name=patch=8_bs=128 \
  --batch_size=128 \
  --n_epochs=200
  