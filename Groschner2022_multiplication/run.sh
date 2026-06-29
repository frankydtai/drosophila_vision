#!/usr/bin/env bash
# Run any command with the project .venv (creates it on first use).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating .venv in $ROOT ..."
  python3 -m venv "$VENV"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$ROOT/requirements.txt"
fi

# Register Jupyter kernel inside the project venv (avoids ~/.local quota).
if [[ ! -d "$VENV/share/jupyter/kernels/groschner2022" ]]; then
  "$PYTHON" -m ipykernel install --prefix="$VENV" --name groschner2022 --display-name "Groschner2022 (.venv)" 2>/dev/null || true
fi

if [[ $# -eq 0 ]]; then
  exec "$PYTHON"
fi

exec "$PYTHON" "$@"
