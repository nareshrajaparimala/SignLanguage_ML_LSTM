#!/bin/bash
cd "$(dirname "$0")"
if [ -d "venv311" ]; then
  source venv311/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi
python3 advanced_mediapipe_server.py
