#!/bin/bash

export MELD_LICENSE='meld_license.txt'

# Use the uv environment (.venv, from setup_uv_env.sh) if it exists, otherwise the conda environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
else
  eval "$(conda shell.bash hook)"
  conda activate meld_graph
fi

if [ $1 = 'pytest' ]; then
  pytest ${@:2}
else
  python scripts/new_patient_pipeline/$1 ${@:2}
fi
