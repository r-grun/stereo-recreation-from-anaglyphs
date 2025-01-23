# Installation
- Create a python environment with anaconda and name `anaconda-gan` <br>
`conda create --name anaglyph-gan python=3.11`
- Install torch, torchaudio, torchvision, cuda support https://pytorch.org/get-started/locally/ <br>
`conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia --yes`
- Install fastai https://docs.fast.ai/ <br>
`conda install -c fastai fastai --yes`
- Install conda requirements from `requirements_conda.txt` <br>
`conda install --yes --file requirements_conda.txt`
- Install pip requirements from `requirements_pip.txt` <br>
`pip install -r requirements_pip.txt`
- Add python kernel to notebook <br>
`python -m ipykernel install --name=anaglyph-gan`