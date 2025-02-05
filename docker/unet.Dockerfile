FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Update package lists
RUN apt-get update

# Install build dependencies
RUN apt-get install -y nano wget

# Download miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install miniconda
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

# create conda environment
RUN conda init bash \
    && . ~/.bashrc \
    && conda create --name anaglyph-unet python=3.12 \
    && conda activate anaglyph-unet \
    && pip install ipython

# Copy requirements files
COPY image_colorization/requirements_conda.txt init/requirements_conda.txt
COPY image_colorization/requirements_pip.txt init/requirements_pip.txt

# Install pytorch
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia --yes

# Install fastai
RUN conda install -c fastai fastai --yes

# Install conda requirements from requirements_conda.txt
RUN conda install --yes --file init/requirements_conda.txt

# Install opencv
RUN pip install opencv-python

# Install pip requirements from `requirements_pip.txt`
RUN pip install -r init/requirements_pip.txt

# Copy directories /data_creation and /image_colorization to the workdir
COPY data_creation /app/data_creation
COPY image_colorization /app/image_colorization

# Add project pythonpath to bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/app/image_colorization" >> ~/.bashrc
RUN . ~/.bashrc

# Set the working directory to /app
WORKDIR /app