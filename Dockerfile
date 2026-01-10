# 1. Base Image: NVIDIA CUDA Runtime
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 2. Install System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Install uv (The Gold Standard for speed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 4. Environment Setup
# Copy strict dependency files
COPY pyproject.toml uv.lock ./

# 5. Install Dependencies
# --system: Installs into the system python (no need for venv inside Docker)
# --frozen: Ensures we stick strictly to the lockfile
RUN uv sync --frozen --system

# 6. Copy Source Code
COPY . .

# 7. Entrypoint
# We can run python directly since packages are in the system path
CMD ["python", "src/train.py", "--help"]
