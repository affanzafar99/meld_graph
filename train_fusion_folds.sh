#!/bin/bash
#SBATCH --job-name=meld_fusion
#SBATCH --partition=A100short
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --array=0-4
#SBATCH --output=meld_fusion_fold%a_%j.log
eval "$(/home/s17azafa/miniconda3/bin/conda shell.bash hook)"
conda activate meld_graph || exit 1
echo "python: $(which python)"
python -c "import meld_graph, torch; print('meld_graph:', meld_graph.__file__, '| GPU:', torch.cuda.is_available())"
cd "$SLURM_SUBMIT_DIR"
python scripts/classifier/train_own_data.py \
    --config_file scripts/config_files/parcel_fusion_fold${SLURM_ARRAY_TASK_ID}.py \
    --demographics_csv /home/s17azafa/meld/demographics_qc_allgroups_noharmo.csv
