#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
