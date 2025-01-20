# DATA
# paths for training
TRAIN_ANAGLYPH_FILE = "../../../data/train_anaglyphs.txt"  # path to the file containing the paths to the anaglyph images
TRAIN_REVERSED_FILE = "../../data/train_reversed.txt" # path to the file containing the paths to the reversed anaglyph images
TRAIN_LEFT_FILE = "../../../data/train_left.txt"  # path to the file containing the paths to the left images (of stereo pairs)
TRAIN_RIGHT_FILE = "../../../data/train_right.txt"  # path to the file containing the paths to the right images (of stereo pairs)

# paths for validation
VALIDATION_ANAGLYPH_FILE = "../../../data/validation_anaglyphs.txt"  # path to the file containing the paths to the anaglyph images
VALIDATION_REVERSED_FILE = "../../data/validation_reversed.txt" # path to the file containing the paths to the reversed anaglyph images
VALIDATION_LEFT_FILE = "../../../data/validation_left.txt"  # path to the file containing the paths to the left images (of stereo pairs)
VALIDATION_RIGHT_FILE = "../../../data/validation_right.txt"  # path to the file containing the paths to the right images (of stereo pairs)

# paths for testing
TEST_ANAGLYPH_FILE = "../../../data/test_anaglyphs.txt"  # path to the file containing the paths to the anaglyph images
TEST_REVERSED_FILE = "../../data/test_reversed.txt" # path to the file containing the paths to the reversed anaglyph images
TEST_LEFT_FILE = "../../../data/test_left.txt"  # path to the file containing the paths to the left images (of stereo pairs)
TEST_RIGHT_FILE = "../../../data/test_right.txt"  # path to the file containing the paths to the right images (of stereo pairs)

# image size
IMAGE_SIZE=256 # size of the image (width = height)

#########################################################################

# TRAINING PARAMETERS
EPOCHS = 20 # number of training epochs
ADAM_LR = 0.0002# Adam optimizer learning rate
ADAM_BETA1 = 0.5 # Adam optimizer beta1

# Storage paths
TEMP_MODEL_PATH = "../models/tmp/"  # path to save temporary models
FINAL_MODEL_PATH = "../models/"  # path to save the trained models

##########################################################################

# Validation
NUM_VALIDATION_IMG = 3  # number of images to store for validation
RESULTS_SAVE_PATH = "../results/"  # path to save the results