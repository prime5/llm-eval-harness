# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install deps in a separate layer so they're cached between code changes.
FROM python:3.11-slim AS builder

WORKDIR /harness

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /harness

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy harness source (excludes .git, __pycache__, .env via .dockerignore)
COPY . .

# Default: run the eval harness and emit JSON to stdout.
# Override EVAL_SUITE via --env or K8s Job env to target a specific suite.
# Example:
#   docker run --rm -e OPENAI_API_KEY=sk-... -e EVAL_SUITE=test_cases/basic_qa.yaml eval-harness
ENTRYPOINT ["python", "run_evals.py"]
CMD ["--format", "json", "--min-pass-rate", "0.0"]
