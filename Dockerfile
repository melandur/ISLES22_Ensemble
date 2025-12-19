# Import image with python environment
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Disable interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install some basic libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    wget \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy sources and dependency files
COPY src/FACTORIZER ./src/FACTORIZER
COPY src/HD-BET ./src/HD-BET
COPY src/NVAUTO ./src/NVAUTO
COPY src/SEALS ./src/SEALS
COPY weights ./weights
COPY pyproject.toml .

# Create a conda environment and install necessary packages
RUN conda create --name isles_ensemble python=3.8.0 pip -y && \
    conda clean -afy

# Activate the environment and install packages
RUN /bin/bash -c "source activate isles_ensemble && \
    pip install uv && \
    uv pip install --no-cache --extra-index-url https://download.pytorch.org/whl/cu113 \
        torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 && \
    uv pip install --no-cache -e ."

# Copy the source code
COPY src/isles22_ensemble.py ./src/isles22_ensemble.py
COPY src/majority_voting.py ./src/majority_voting.py
COPY src/utils.py ./src/utils.py
COPY src/__init__.py ./src/__init__.py
COPY main.py .

# Set PYTHONPATH for local modules (SEALS, FACTORIZER, HD-BET)
ENV PYTHONPATH="${PYTHONPATH}:/app/src/SEALS:/app/src/FACTORIZER/model/factorizer:/app/src/HD-BET"

# Run docker will start the main.py
ENTRYPOINT ["/bin/bash", "-c", "source activate isles_ensemble && python main.py \"$@\"", "--"]
