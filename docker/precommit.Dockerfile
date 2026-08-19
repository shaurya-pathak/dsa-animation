FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        ffmpeg \
        libcairo2-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip uv

# Keep Cairo in the private-beta render image on the same family that the
# self-contained web preview embeds. The font asset is licensed alongside it.
COPY src/kaivra/assets/fonts/InterVariable.ttf /usr/local/share/fonts/kaivra/InterVariable.ttf
RUN fc-cache -f

WORKDIR /workspace
