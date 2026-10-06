FROM python:3.13-slim

# Hugging Face Spaces run containers as user 1000, not as root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    NUMBA_CACHE_DIR=/home/user/.cache/numba \
    VSEARCH_DATA=/home/user/app/data
WORKDIR /home/user/app

# CPU-only PyTorch is far smaller than the default build, which bundles GPU libraries.
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user vsearch vsearch
COPY --chown=user app app
COPY --chown=user data data

# Start the service once while building: this downloads the embedding model
# and compiles the Numba kernels into the image, so the container starts fast.
RUN python -c "from pathlib import Path; from app.service import SearchService; SearchService(Path('data'))"
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]