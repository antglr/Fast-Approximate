#!/bin/bash
# Any script of this repo on one GB200: sbatch slurm/run.sbatch scripts/regime_bench.py results/datasets results/regime_bench
#SBATCH --job-name=fa_adnan
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --time=02:00:00
source slurm/env.sh
mkdir -p slurm_logs
srun apptainer exec --nv $SIF python3 -u "$@"
