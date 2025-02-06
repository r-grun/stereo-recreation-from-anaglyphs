# Data Creation
> This folder contains the code to create the dataset and prepare data for training.

## Data Creation
The training, the stereo images need to be sliced into left and right images, then sliced into smaller ones of the size 256 x 256 pixels. 
Additionally, the corresponding anaglyph image for each slice is created, as well as its inverse anaglyph image.
The inverse anaglyph is used as the ground truth for the training, as it is the predicted value of the model.

`create_anaglyphs_batch.py` contains the code to create necessary data.
`create_anaglyphs_batch_gpu` does the same as `create_anaglyphs_batch.py` but utilizes the GPU for faster processing.
`create_txt_from_csvs.py` creates the necessary text files for training the model, which contain the locations of the test, train and validation images.

_Before running these files, make sure to use the correct config parameters in the beginning of the script._

---

Create all necessary data for training the model by running the following command:
```bash
python create_anaglyphs_batch.py
```

Create all necessary data for training the model using the GPU by running the following command:
```bash
python create_anaglyphs_batch_gpu.py
```

Create the necessary text files for training the model by running the following command:
```bash
python create_txt_from_csvs.py
```