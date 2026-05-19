FROM python:3.10-slim

# Install system dependencies required for OpenCV (Modern version)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up user permissions for Hugging Face Spaces (UID 1000 is required)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 3. Copie et installation des packages Python
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copie de tout le reste de ton projet Sahl Express
COPY --chown=user . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
