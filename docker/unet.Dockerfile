FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Update package lists
RUN apt-get update

# Install nano editor
RUN apt-get install -y nano

# Install miniconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
RUN source ~/miniconda3/bin/activate
RUN conda init --all

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
RUN pip install -y opencv-python

# Install pip requirements from `requirements_pip.txt`
RUN pip install -r init/requirements_pip.txt

# Copy directories /data_creation and /image_colorization to the workdir
COPY data_creation /app/data_creation
COPY image_colorization /app/image_colorization

# Add project pythonpath to bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/app/image_colorization" >> ~/.bashrc
RUN source ~/.bashrc

# Set the working directory to /app
WORKDIR /app