# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install Python and deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libsndfile1 \
    libsamplerate0-dev \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Rest is same...
RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]