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

This command will process the input anaglyph image using the specified U-Net model, save the reversed anaglyph image, and create a stereo pair image with the specified dimensions.

To run the infer_unet.py script, use the following command:<br>
```python image_colorization/src/infer_model.py --model_path <model_path> --image_path <image_path> --output_path <output_path> --stereo_output_path <stereo_output_path> [--img_height <img_height>] [--img_width <img_width>] [--output_height <output_height>] [--output_width <output_width>]```

Arguments:<br>

- `--model_path` (required): Path to the U-Net model.
- `--image_path` (required): Path to the input anaglyph image.
- `--output_path` (required): Path to save the reversed anaglyph image.
- `--stereo_output_path` (required): Path to save the stereo pair image.
- `--model_input_height` (optional): Height of the input of the model. Input image will be transformed to this value. Default is 256.
- `--model_input_width` (optional): Width of the input of the model. Input image will be transformed to this value Default is 256.
- `--output_height`(optional): Height of the output reversed anaglyph image and stereo-pair. Default is 256.
- `--output_width` (optional): Width of the output reversed anaglyph image. The stereo-pair's width is `2 * output_width`. Default is 256.


Example:<br>
```python image_colorization/src/infer_model.py --model_path path/to/unet_checkpoint.pth --image_path path/to/anaglyph_image.png --output_path path/to/output_image.png --stereo_output_path path/to/stereo_pair.png --img_height 256 --img_width 256 --output_height 256 --output_width 256```
