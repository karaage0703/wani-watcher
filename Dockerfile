# NVIDIA CUDA 11.8 with PyTorch support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy only dependency files for caching
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies using uv with limited concurrency
# RUN uv venv
# ENV UV_CONCURRENT_DOWNLOADS=1
RUN uv sync

# Create necessary directories
RUN mkdir -p models runs training_data/images training_data/labels

# Set environment variables for GPU support
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["uv", "run", "python", "train_wani_detector.py", "--epochs", "50", "--batch-size", "8"]
