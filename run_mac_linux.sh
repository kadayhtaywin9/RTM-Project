#!/usr/bin/env bash
set -e
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -c "import streamlit, xgboost, geopandas, rasterio" 2>/dev/null || python -m pip install -r requirements.txt
python -m streamlit run app.py
