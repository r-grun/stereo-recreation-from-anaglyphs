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


---

# Usage

Note: the following commands are displayed for U-Net model as it is the current model under active development. For GAN model, replace `unet` with `gan` in the commands.

## Training
### In Jupyter Notebook
- Open `recreate_anaglyphs_unet.ipynb` for U-Net training in Jupyter Notebook

### In Terminal
- Run `python .src/train_unet.py` for U-Net training in terminal


## Visualizing training metrics
- Open `plot_training_results_unet.ipynb` for U-Net training metrics visualization in Jupyter Notebook.<br>
Make sure to select the corresponding csv file to display its contents


## Testing/Inference
Interference is currently only supported for U-Net. GAN interference is not yet implemented.

### In Jupyter Notebook
- Open `infer_unet.ipynb` for U-Net  inference in Jupyter Notebook

### In Terminal
- Run `python .src/infer_unet.py` for U-Net inference in terminal