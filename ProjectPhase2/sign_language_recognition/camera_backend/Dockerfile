FROM python:3.10-slim

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_advanced.txt .
RUN pip install --no-cache-dir -r requirements_advanced.txt
RUN pip install pyserial

# Copy the rest of the backend code
COPY . .

# Run the advanced server
CMD ["python", "advanced_mediapipe_server.py"]
