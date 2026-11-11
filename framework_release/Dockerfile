# CPU Phase A with externally supplied, exact native visibility inputs.
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/framework_release
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/opt/framework_release:/opt/framework_release/src_final \
    MULTIOBJ3DGS_PROJECT_ROOT=/opt/framework_release \
    PYTHONDONTWRITEBYTECODE=1
CMD ["python", "-m", "experiments.streaming_evaluation.two_phase", "--help"]
