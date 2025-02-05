import torch
import config_notebook as c
import config_notebook_test_run as test_c
import datetime
from src.unet import UNet
from src.dataloader_anaglyph_reversed import make_dataloaders
from src.train_unet import set_global_config
from src.train_unet import train_unet
import config
import config_test_run
import argparse

def main(is_test_run=False):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Initialize models
    unet = UNet()

    # Choose config based on is_test_run
    if is_test_run:
        current_config = config_test_run
    else:
        current_config = config

    # Set train config
    set_global_config(current_config)

    # Create timestamp for training runs
    training_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Started training at {training_run_timestamp}")

    if is_test_run:
        print("Running test run")

        # Make dataloaders for single item
        single_item_dl = make_dataloaders(path_anaglyph=c.TRAIN_ANAGLYPH_FILE, path_reversed=c.TRAIN_REVERSED_FILE, files_limit=1)
        print(f"Size of single item dataloader: {len(single_item_dl)}")

        # Run training
        train_unet(model=unet, train_dl=single_item_dl, val_dl=single_item_dl, device=device, timestamp=training_run_timestamp)
    else :
        print("Running full training")

        # Make dataloaders
        training_dl = make_dataloaders(path_anaglyph=current_config.TRAIN_ANAGLYPH_FILE, path_reversed=current_config.TRAIN_REVERSED_FILE)
        validation_dl = make_dataloaders(path_anaglyph=current_config.VALIDATION_ANAGLYPH_FILE, path_reversed=current_config.VALIDATION_REVERSED_FILE)
        print(f"Size of training dataloader: {len(training_dl)}, Size of validation dataloader: {len(validation_dl)}")

        # Run training
        train_unet(model=unet, train_dl=training_dl, val_dl=validation_dl, device=device, timestamp=training_run_timestamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet training")
    parser.add_argument('--is_test_run', action='store_true', help="Set this flag for a test run")
    args = parser.parse_args()
    main(is_test_run=args.is_test_run)