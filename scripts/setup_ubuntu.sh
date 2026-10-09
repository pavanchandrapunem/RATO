#!/usr/bin/env bash
# Execute from RATO repo root after installing Miniconda/Miniforge; safe to inspect first.
set -euo pipefail
if ! command -v conda >/dev/null 2>&1; then
  echo 'Install Miniforge/Miniconda first, then run this script.' >&2; exit 1
fi
conda create -n rato python=3.11.5 -y
# conda run avoids assumptions about shell activation
conda run -n rato python -m pip install --upgrade pip
conda run -n rato python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
conda run -n rato python -m pip install -r requirements.txt
conda run -n rato python -m pip install -e .
conda run -n rato python -m pytest -q
echo 'Activate with: conda activate rato'
