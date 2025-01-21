import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.optim import Adam
from torchvision.utils import save_image
import datetime
import csv
import torch.cuda.amp as amp
import torchvision.transforms as transforms

# global parameters
loss_function = nn.CrossEntropyLoss()

def set_global_config(config_module):
    global c
    c = config_module

def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def display_validation_images(model, validation_dl, device, results_save_path, epoch):
    """
    Display the validation images: anaglyph, original reversed, and generated reversed.
    """
    model.eval()  # Set to evaluation mode

    # Get the transforms from the dataset
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dl):
            if batch_idx >= c.NUM_VALIDATION_IMG:
                break

            img_anaglyph = batch['a'].to(device)  # Preprocessed anaglyph image
            img_reversed = batch['r'].to(device)

            # Forward pass
            generated_reversed = model(img_anaglyph)

            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save the results
            save_image(generated_reversed, f"{results_save_path}/unet_epoch_{epoch+1}_img_{batch_idx+1}_{timestamp}_reversed.png")
            print(f"Saved validation results for image {batch_idx+1} of epoch {epoch+1} at {timestamp}")

            # Denormalize and convert to PIL images
            img_anaglyph = denormalize(img_anaglyph.cpu(), mean, std)
            img_reversed = denormalize(img_reversed.cpu(), mean, std)
            generated_reversed = denormalize(generated_reversed.cpu(), mean, std)

            to_pil = transforms.ToPILImage()

            # Plot the results
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(to_pil(img_anaglyph[0]))
            axes[0].set_title("Anaglyph")
            axes[0].axis("off")

            axes[1].imshow(to_pil(img_reversed[0]))
            axes[1].set_title("Original Reversed")
            axes[1].axis("off")

            axes[2].imshow(to_pil(generated_reversed[0]))
            axes[2].set_title("Generated Reversed")
            axes[2].axis("off")

            plt.show()

    model.train()  # Return to training mode


def calculate_validation_loss(model, validation_dl, device, loss_function):
    """
    Calculate the validation loss for the model.
    """
    model.eval()  # Set to evaluation mode
    val_loss = 0

    with torch.no_grad():
        for batch in validation_dl:
            img_anaglyph = batch['a'].to(device)  # Preprocessed anaglyph image
            img_reversed = batch['r'].to(device)

            # Forward pass
            generated_reversed = model(img_anaglyph)
            loss = loss_function(generated_reversed, img_reversed)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_dl)
    model.train()  # Return to training mode

    return avg_val_loss


def train_unet(model, train_dl, val_dl, num_epochs, device, timestamp, test_run=False):
    """
    Train the U-Net model with optimizations.
    """

    # Move model to the specified device
    model = model.to(device)

    # Define the optimizer (e.g., Adam)
    optimizer = Adam(model.parameters(), lr=c.ADAM_LR)

    # Prepare to save losses to a CSV file
    losses_csv_path = os.path.join(c.RESULTS_PATH if not test_run else c.TEST_RESULTS_PATH, f"training_losses_unet_{timestamp}.csv")
    with open(losses_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    # Initialize the GradScaler for mixed precision training
    scaler = amp.GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0

        loop = tqdm(train_dl, leave=True)
        for batch_idx, batch in enumerate(loop):
            # Extract anaglyph and reversed images from the batch
            anaglyph = batch['a'].to(device, non_blocking=True)
            reversed_image = batch['r'].to(device, non_blocking=True)

            # Forward pass with mixed precision
            with amp.autocast():
                outputs = model(anaglyph)
                loss = loss_function(outputs, reversed_image)

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update progress bar
            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        # Log epoch loss
        avg_epoch_loss = train_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Calculate validation loss
        avg_val_loss = calculate_validation_loss(model, val_dl, device, loss_function)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Save losses to CSV
        with open(losses_csv_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch + 1, avg_epoch_loss, avg_val_loss])

        # Perform validation and save model/checkpoint only every num_store_every epochs
        if test_run or ((epoch + 1) % c.NUM_STORE_EVERY == 0) or ((epoch + 1) == num_epochs):
            display_validation_images(model=model, validation_dl=val_dl, device=device, results_save_path=c.RESULTS_PATH if not test_run else c.TEST_RESULTS_PATH, epoch=epoch)

            # Save model checkpoint
            checkpoint_path = os.path.join(c.MODEL_PATH if not test_run else c.TEST_MODEL_PATH, f"unet_checkpoint_{timestamp}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")