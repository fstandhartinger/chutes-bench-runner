FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the exact dependency set resolved from requirements.txt.  The
# checksum manifest makes source/lock drift a build error instead of silently
# invoking pip's resolver again.
COPY requirements.txt requirements.lock requirements.lock.sha256 ./
RUN sha256sum --check requirements.lock.sha256 \
    && pip install --no-cache-dir --require-hashes -r requirements.lock

ARG BENCH_RUNNER_GIT_SHA=""
ENV BENCH_RUNNER_GIT_SHA=${BENCH_RUNNER_GIT_SHA}
LABEL ai.chutes.bench-runner.git-sha=${BENCH_RUNNER_GIT_SHA}

# Copy application
COPY . .

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]


























