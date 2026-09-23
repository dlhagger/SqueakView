FROM nvcr.io/nvidia/deepstream:9.1-triton-multiarch

USER root

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-keysyms1 \
        pkg-config \
        sudo && \
    rm -rf /var/lib/apt/lists/*

RUN usermod --shell /bin/bash triton-server && \
    install -d -m 0755 -o 1000 -g 1000 \
        /home/triton-server \
        /home/triton-server/.cache \
        /home/triton-server/.config && \
    chown -R 1000:1000 /home/triton-server && \
    echo "triton-server ALL=(ALL) NOPASSWD:ALL" \
        > /etc/sudoers.d/triton-server && \
    chmod 0440 /etc/sudoers.d/triton-server

RUN curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR=/usr/local/bin sh

ENV HOME=/home/triton-server

USER triton-server