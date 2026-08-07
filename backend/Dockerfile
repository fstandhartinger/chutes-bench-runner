FROM python:3.11-slim

ARG BENCH_RUNNER_GIT_SHA=""
ENV BENCH_RUNNER_GIT_SHA=${BENCH_RUNNER_GIT_SHA}
LABEL ai.chutes.bench-runner.git-sha=${BENCH_RUNNER_GIT_SHA}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir -e ".[benchmarks]"

# Copy application
COPY . .

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]




























