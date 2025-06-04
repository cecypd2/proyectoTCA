#!/bin/bash

# Iniciar FastAPI en background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Espera breve y lanza Streamlit
sleep 5
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0