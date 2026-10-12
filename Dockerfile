# Render-only preparation + CPU inference. No weights or secrets in build layers.
FROM python:3.12-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements-api.txt requirements-cpu.txt ./
ENV CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DLLAMA_CURL=OFF" CMAKE_BUILD_PARALLEL_LEVEL=2
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-cpu.txt
RUN git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp && git checkout 4227c9be4268ac844921b90f31595f81236bd317 \
    && cmake -B build -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release --target llama-quantize -j 2

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
WORKDIR /app
COPY --from=build /opt/llama.cpp /opt/llama.cpp
COPY requirements-api.txt requirements-cpu.txt requirements-convert.txt ./
# CPU torch is needed only in the finite conversion subprocess, never for serving.
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-convert.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    NATLAS_BACKEND=gguf NATLAS_ENABLED=true NATLAS_ARTIFACT_DIR=/data/natlas/model \
    NATLAS_MANAGED_ARTIFACT=true NATLAS_SETUP_MODE=prepare HF_LOCAL_FILES_ONLY=true \
    NATLAS_PREPARE_WORK_DIR=/data/natlas/work NATLAS_PREPARE_TIMEOUT_SECONDS=21600 \
    NATLAS_CONTEXT_SIZE=2048 NATLAS_MAX_INPUT_TOKENS=1920 NATLAS_MAX_NEW_TOKENS=128 \
    NATLAS_THREADS=2 NATLAS_BATCH_SIZE=128 NATLAS_TIMEOUT_SECONDS=120 PORT=10000
COPY app ./app
COPY scripts/serve.py scripts/smoke_cpu.py scripts/convert_model.py ./scripts/
EXPOSE 10000
CMD ["python", "scripts/serve.py"]
