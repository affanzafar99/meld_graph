#!/bin/bash
#SBATCH --job-name=meld_eval
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --output=meld_eval_%j.log
eval "$(/home/s17azafa/miniconda3/bin/conda shell.bash hook)"
conda activate meld_graph || exit 1
cd "$SLURM_SUBMIT_DIR"
python scripts/classifier/evaluate_parcel_fusion.py "$@" \
    --demographics_csv /home/s17azafa/meld/meld_graph_private/data/my_data/demographics_all_meld.csv
