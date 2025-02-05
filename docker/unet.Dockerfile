FROM continuumio/miniconda3:23.5.2-0-alpine

# Install nano editor
RUN apk add nano

# Copy requirements files
COPY image_colorization/requirements_conda.txt init/requirements_conda.txt
COPY image_colorization/requirements_pip.txt init/requirements_pip.txt

# Install pytorch
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia --yes

# Install fastai
RUN conda install -c fastai fastai --yes

# Install conda requirements from requirements_conda.txt
RUN conda install --yes --file init/requirements_conda.txt

# Install pip requirements from `requirements_pip.txt`
RUN pip install -r init/requirements_pip.txt

# Copy directories /data_creation and /image_colorization to the workdir
COPY data_creation /app/data_creation
COPY image_colorization /app/image_colorization

# Set the working directory to /app
WORKDIR /app