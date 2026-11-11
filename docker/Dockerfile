# CPU environment for LeRobot + LIBERO. Software rendering through OSMesa.
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git pkg-config ffmpeg \
        libosmesa6-dev libgl1 libglu1-mesa libegl1-mesa-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

# Versions used for all reported results (checked in the built image).
RUN pip install --no-cache-dir "lerobot[smolvla,libero]==0.7.0" "torch==2.15.0" "mujoco==3.8.1" "robosuite==1.4.0"
RUN pip install --no-cache-dir pytest ruff lpips scikit-image h5py pandas pyarrow matplotlib scikit-learn

ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    HF_HOME=/cache/huggingface \
    LIBERO_CONFIG_PATH=/cache/libero

# LIBERO checks this path for assets on import. Keep them in the cache volume.
RUN mkdir -p /root/.cache && ln -s /cache/libero /root/.cache/libero

WORKDIR /workspace
