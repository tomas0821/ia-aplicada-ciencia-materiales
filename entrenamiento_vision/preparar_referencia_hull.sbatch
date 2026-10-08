#!/bin/bash
#SBATCH --job-name=ref_hull
#SBATCH --partition=nukwa-l40s
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=ref_hull_%j.out
cd "$SLURM_SUBMIT_DIR"
/opt/python/mamba3/envs/python-AI-materials/bin/python entrenamiento_vision/preparar_referencia_hull.py datos/hull
