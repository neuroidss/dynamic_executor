#!/bin/bash

 echo "Installing python dependencies..."
# echo "Installing Node.js dependencies..."
# npm install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

 echo ""
 echo "Setup complete. Make sure you have:"
 echo "1. Ollama running (e.g., 'ollama serve')"
 echo "2. The qwen2.5-coder:7b-instruct-q8_0 model pulled ('ollama pull qwen2.5-coder:7b-instruct-q8_0')"
 echo "3. The nomic-embed-text model pulled ('ollama pull nomic-embed-text')"
 echo "4. ChromaDB running (e.g., via Docker or 'chroma run --path ./chroma_db_data')"
 echo ""
 echo "You might need to adjust ChromaDB host/port and Ollama host/port in the scripts if not using defaults."
 echo ""
 echo "To run:"
 echo "./run.sh"
 echo ""
