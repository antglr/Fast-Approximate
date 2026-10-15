#!/bin/bash
# Jupyter kernel on one GB200 for local VS Code: sbatch slurm/jupyter.sbatch, then see slurm_logs/fa_jupyter_<id>.out
#SBATCH --job-name=fa_jupyter
#SBATCH --reservation=pdl18409_oct2026
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --time=08:00:00
source slurm/env.sh
mkdir -p slurm_logs
PORT=${PORT:-8893}
echo "node: $(hostname)  port: $PORT"
echo "tunnel: ssh -N -L $PORT:$(hostname):$PORT dalia"
srun apptainer exec --nv --env PYTHONPATH=$HOME/pyext $SIF \
    jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --notebook-dir=$HOME/Fast-Approximate
