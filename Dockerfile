# ai-cluster/chunking/Dockerfile
#
# Standalone chunker FastAPI service. Heavy ML deps (docling, surya-ocr,
# torch, transformers, opencv) live ONLY in this image — the AI Gateway
# image stays thin (Phase A3 split).
#
# Listens on 8100 to match the AI Gateway's CHUNKER_URL default.

FROM python:3.12-slim

WORKDIR /app

# 1) Install heavyweight ML deps + pin opencv-python-headless to a known-good
#    version (the requirements.txt entry pulls in 4.11 but docling is happier
#    with 4.13).
#    --extra-index-url is required so pip resolves the +cu126 torch wheels.
#    cu126 (not cu124, which is what the host Raas conda env uses) because
#    surya-ocr 0.17 / docling-surya 0.1 require torch>=2.7 and PyTorch
#    stopped publishing cu124 wheels at torch 2.6. cu126 wheels remain
#    forward-compatible with CUDA 12.4 drivers via NVIDIA minor-version
#    compatibility. Falls back to CPU at runtime if the container is
#    launched without nvidia-container-toolkit / --gpus all.
COPY requirements.txt .
RUN pip install --default-timeout=1000 \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r requirements.txt && \
    pip uninstall -y opencv-python-headless && \
    pip install --no-cache-dir "opencv-python-headless==4.13.0.92" && \
    python -c "import cv2; print('cv2 OK:', cv2.__version__)" && \
    python -c "import torch; print(f'torch={torch.__version__}, cuda_available={torch.cuda.is_available()}')"

# 2) Copy source. shared/ is mounted by docker-compose (or via PYTHONPATH for
#    native dev) so the /predict endpoint can import the gateway contract.
COPY src/ ./src

# 3) Install the docling-custom-ocr plugin from the local package so the
#    OCR_BACKEND=custom_api path can find it via the docling factory.
#    OCR_BACKEND=surya is the safer default for local dev (no external
#    dependency on the OCR HTTP service); keep custom_api opt-in.
RUN pip install --no-cache-dir -e ./src/docling-custom-ocr

ENV PYTHONPATH=/app
ENV DOCLING_ALLOW_EXTERNAL_PLUGINS=true
ENV OCR_BACKEND=surya

EXPOSE 8100

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8100"]