#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "ERROR: Run bash run_mac_linux.sh first to set up .venv." >&2
  exit 1
fi
.venv/bin/python scripts/check_runtime.py
exec .venv/bin/python -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
