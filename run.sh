#!/bin/bash

 # Optional: Start ChromaDB if not running elsewhere
 # echo "Starting ChromaDB..."
 # chroma run --path ./chroma_db_data & # Run in background
 # CHROMA_PID=$!
 # sleep 5 # Give ChromaDB time to start

 echo "Starting python server..."
# echo "Starting Node.js server..."
# node server.js
source venv/bin/activate
 python server.py "$@"

 # Optional: Stop ChromaDB when server exits
 # echo "Stopping ChromaDB..."
 # kill $CHROMA_PID
