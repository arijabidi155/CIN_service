FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user permissions for Hugging Face Spaces (UID 1000 is required)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY --chown=user . .

# Warm up the YOLOv11 model download at build time to avoid delays on first API call
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Start the server on port 7860 (Hugging Face Spaces default port)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
