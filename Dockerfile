# CPU inference only. Convert N-ATLaS Q4_K_M off Render and publish the
# verified artifact to a pinned private artifact repository before deployment.
FROM python:3.12-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements-cpu.txt ./
ENV CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DLLAMA_CURL=OFF" CMAKE_BUILD_PARALLEL_LEVEL=2
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-cpu.txt

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
WORKDIR /app
COPY requirements-cpu.txt ./
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    NATLAS_BACKEND=gguf NATLAS_ENABLED=true NATLAS_ARTIFACT_DIR=/data/natlas/model \
    NATLAS_MANAGED_ARTIFACT=false NATLAS_REQUIRE_VERIFIED_ARTIFACT=true NATLAS_SETUP_MODE=serve HF_LOCAL_FILES_ONLY=false \
    NATLAS_CONTEXT_SIZE=2048 NATLAS_MAX_INPUT_TOKENS=1792 NATLAS_MAX_NEW_TOKENS=128 \
    NATLAS_THREADS=4 NATLAS_BATCH_SIZE=64 NATLAS_TIMEOUT_SECONDS=180 PORT=10000
RUN useradd --create-home --uid 10001 ai && mkdir -p /data/natlas/model /data/huggingface \
    && chown -R ai:ai /data /app
COPY --chown=ai:ai app ./app
COPY --chown=ai:ai scripts/serve.py scripts/smoke_cpu.py ./scripts/
EXPOSE 10000
CMD ["python", "scripts/serve.py"]
