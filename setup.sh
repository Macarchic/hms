#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:-python3}
VENV=.venv

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:"
echo "  source .venv/bin/activate"
