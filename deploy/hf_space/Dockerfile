# Hugging Face Space (Docker SDK, free CPU hardware). Built from the folder made by
# scripts/build_hf_space.py. The model is downloaded once at build time (pinned revision),
# so the running Space never downloads it again.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MM_DEVICE=cpu \
    MM_HOST=0.0.0.0 \
    PORT=7860 \
    MM_BUSY_WAIT_S=240 \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Bake the pinned model revision into the image.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775')"
ENV HF_HUB_OFFLINE=1

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "scripts/run_web.py", "--no-browser"]
