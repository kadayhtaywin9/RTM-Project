#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

compatible='import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) and sys.maxsize > 2**32 else 1)'
if [ ! -x .venv/bin/python ]; then
  if [ -e .venv ]; then
    echo "ERROR: .venv is incomplete. Repair or rename it before retrying." >&2
    exit 1
  fi
  interpreter=""
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$compatible"; then
      interpreter="$candidate"
      break
    fi
  done
  if [ -z "$interpreter" ]; then
    echo "ERROR: Install 64-bit Python 3.11 or 3.12, then retry." >&2
    exit 1
  fi
  "$interpreter" -m venv .venv
fi
if ! .venv/bin/python -c "$compatible"; then
  echo "ERROR: .venv requires 64-bit Python 3.11 or 3.12. Repair or rename it." >&2
  exit 1
fi
if ! .venv/bin/python scripts/check_runtime.py; then
  .venv/bin/python -m pip install -r requirements.txt
  .venv/bin/python scripts/check_runtime.py
fi
exec .venv/bin/python -m streamlit run app.py
