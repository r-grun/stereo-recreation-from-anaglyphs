import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from src.unet import UNet
from src.dataloader_anaglyph_reversed import make_dataloaders
from src.train_unet import set_global_config, display_current_config_parameters, train_unet
import src.config as config
import src.config_test_run as config_test_run
import argparse
from tqdm import tqdm

def main(test_run=True):
    # Choose config based on is_test_run
    if test_run:
        current_config = config_test_run
    else:
        current_config = config

    # Create timestamp for training runs
    training_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Started training at {training_run_timestamp}")

    # Init writer for TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(current_config.LOGS_PATH, f"{training_run_timestamp}"))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Initialize models
    unet = UNet()

    # Use DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Set train config
    set_global_config(current_config)
    display_current_config_parameters(store=(not test_run), file_path= os.path.join(current_config.MODEL_PATH, f"{training_run_timestamp}_{current_config.STORE_CONFIG_NAME}"))

    try:
        # Log hyperparameters to TensorBoard
        hparams = {k: v for k, v in vars(current_config).items() if not k.startswith('__') and not callable(v)}
        writer.add_hparams(hparams, {})

        if test_run:
            print("Running test run")

            # Make dataloaders for single item
            single_item_dl = make_dataloaders(batch_size=current_config.DATALOADER_BATCH_SIZE, n_workers=current_config.DATALOADER_N_WORKERS, path_anaglyph=current_config.TRAIN_ANAGLYPH_FILE, path_reversed=current_config.TRAIN_REVERSED_FILE, files_limit=3)
            print(f"Size of single item dataloader: {len(single_item_dl)}")

            # Run training
            train_unet(model=unet, train_dl=single_item_dl, val_dl=single_item_dl, device=device, timestamp=training_run_timestamp, tqdm=tqdm, writer=writer)
        else:
            print("Running full training")

            # Make dataloaders
            training_dl = make_dataloaders(batch_size=current_config.DATALOADER_BATCH_SIZE, n_workers=current_config.DATALOADER_N_WORKERS, path_anaglyph=current_config.TRAIN_ANAGLYPH_FILE, path_reversed=current_config.TRAIN_REVERSED_FILE)
            validation_dl = make_dataloaders(batch_size=current_config.DATALOADER_BATCH_SIZE, n_workers=current_config.DATALOADER_N_WORKERS, path_anaglyph=current_config.VALIDATION_ANAGLYPH_FILE, path_reversed=current_config.VALIDATION_REVERSED_FILE)
            print(f"Size of training dataloader: {len(training_dl)}, Size of validation dataloader: {len(validation_dl)}")

            # Run training
            train_unet(model=unet, train_dl=training_dl, val_dl=validation_dl, device=device, timestamp=training_run_timestamp, tqdm=tqdm, writer=writer)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        writer.close()
        print("Writer closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet training")
    parser.add_argument('--test_run', action='store_true', help="Set this flag for a test run")
    args = parser.parse_args()
    main(test_run=args.test_run)
    print("Training completed.")