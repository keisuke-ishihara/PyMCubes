#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -m venv "${SCRIPT_DIR}/venv"
source "${SCRIPT_DIR}/venv/bin/activate"
pip install --upgrade pip
pip install setuptools wheel Cython "numpy~=2.0"   # build deps from pyproject.toml
pip install -e "${SCRIPT_DIR}"                      # PyMCubes from source
pip install "trimesh>=4.0" "matplotlib>=3.7"

echo "Done. Activate: source venv/bin/activate && python convergence_test.py"
