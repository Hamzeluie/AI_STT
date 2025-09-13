# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install Python and deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libsndfile1 \
    libsamplerate0-dev \
    gcc \
    g++ \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Rest is same...
RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-root -v

COPY . .

EXPOSE 8000

# Copy the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh

# Make it executable
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["poetry", "run", "python", "main.py"]
