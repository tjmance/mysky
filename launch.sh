#!/bin/bash
# AI Video Generation Studio Launch Script

echo "🎬 AI Video Generation Studio"
echo "Starting the application..."

# Check Python version
python3 --version

# Create outputs directory if it doesn't exist
mkdir -p outputs models temp uploads

# Launch Streamlit
echo "Launching Streamlit on http://localhost:8501"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
