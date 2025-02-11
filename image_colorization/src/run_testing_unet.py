import torch
import datetime
from src.unet import UNet
from src.dataloader_anaglyph_reversed import make_dataloaders
import src.config as config
import argparse
from src.test_model import set_global_config, test_model

def main(model_path=None):

    # Create timestamp for testing runs
    testing_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Started testing at {testing_run_timestamp}")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Use DataParallel for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Initialize models
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set global config
    set_global_config(config)

    try:
        print("Running full training")

        # Make dataloaders
        testing_dl = make_dataloaders(batch_size=config.DATALOADER_BATCH_SIZE, n_workers=config.DATALOADER_N_WORKERS, path_anaglyph=config.TEST_ANAGLYPH_FILE, path_reversed=config.TEST_REVERSED_FILE)
        print(f"Size of testing dataloader: {len(testing_dl)}")

        # Run training
        test_model(model=model, test_dl=testing_dl, device=device, timestamp=testing_run_timestamp)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet testing")
    parser.add_argument('--model_path', type='str', help="Path to the model to test")
    args = parser.parse_args()
    main(model_path=args.test_run)
    print("Testing completed.")