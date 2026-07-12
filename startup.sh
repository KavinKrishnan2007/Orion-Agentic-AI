#!/bin/bash
set -e

exec python -m streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8000}" \
  --server.headless=true \
  --browser.gatherUsageStats=false
