FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Update package lists
RUN apt-get update

# Install build dependencies
RUN apt-get install -y nano python3.10 python3-pip python3-dev python3-venv build-essential
RUN alias python=python3

# Copy requirements files
COPY image_colorization/requirements_conda.txt init/requirements_conda.txt
COPY image_colorization/requirements_pip.txt init/requirements_pip.txt

# Install pytorch
RUN pip install torch torchvision torchaudio

# Install conda requirements from requirements_conda.txt
RUN pip install -r init/requirements_conda.txt && pip install -r init/requirements_pip.txt

# Copy directories /data_creation and /image_colorization to the workdir
COPY data_creation /app/data_creation
COPY image_colorization /app/image_colorization

# Add project pythonpath to bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/app/image_colorization" >> ~/.bashrc
RUN . ~/.bashrc

# Set the working directory to /app
WORKDIR /app
RUN alias python=python3

EXPOSE 6006 8888