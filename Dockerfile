ARG BASE=nvcr.io/nvidia/pytorch:23.10-py3

FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_NO_TORCHVISION=1
ENV TRANSFORMERS_NO_VISION=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/voice

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchaudio==2.1.0
RUN pip install --no-cache-dir --no-deps git+https://github.com/ysharma3501/NovaSR.git

COPY . /srv/voice

ENV NOVASR_DEVICE=auto
ENV NOVASR_HALF=1
ENV FFMPEG_PATH=ffmpeg

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
