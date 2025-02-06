import os
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision.utils import save_image
import csv
import torchvision.transforms as transforms
from src.image_loss import ImageLoss
from torch.utils.tensorboard import SummaryWriter

def set_global_config(config_module):
    """ Set the global configuration."""
    global c
    c = config_module

def display_current_config_parameters(store=False, file_path=None):
    """ Display the current configuration parameters and store them in a file if store is True."""
    config_parameters = []
    for attribute in dir(c):
        if not attribute.startswith("__"):
            param = f"{attribute}: {getattr(c, attribute)}"
            print(param)
            config_parameters.append(param)

    if store:
        with open(file_path, 'w') as file:
            for param in config_parameters:
                file.write(param + '\n')
        print(f"Configuration parameters stored in {file_path}")

def denormalize(tensor, mean, std):
    """ Denormalize a tensor with the given mean and standard deviation."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def display_validation_images(img_anaglyph, img_reversed, generated_reversed):
    """ Display the anaglyph, original reversed and generated reversed images."""
    to_pil = transforms.ToPILImage()

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

def store_validation_images(model, validation_dl, device, epoch, timestamp):
    """ Store the anaglyph, original reversed and generated reversed images."""
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_dl):
            if batch_idx >= c.NUM_VALIDATION_IMG:
                break

            img_anaglyph = batch['a'].to(device)
            img_reversed = batch['r'].to(device)
            generated_reversed = model(img_anaglyph)

            if c.STORE_VALIDATION_IMGS:
                for img_idx in range(img_anaglyph.size(0)):
                    save_image(img_anaglyph[img_idx], f"{c.RESULTS_PATH}/unet_epoch_{epoch+1}_batch_{batch_idx+1}_img_{img_idx+1}_{timestamp}_anaglyph.png")
                    save_image(img_reversed[img_idx], f"{c.RESULTS_PATH}/unet_epoch_{epoch+1}_batch_{batch_idx+1}_img_{img_idx+1}_{timestamp}_reversed.png")
                    save_image(generated_reversed[img_idx], f"{c.RESULTS_PATH}/unet_epoch_{epoch+1}_batch_{batch_idx+1}_img_{img_idx+1}_{timestamp}_generated_reversed.png")
                    print(f"Saved validation results for image {img_idx+1} of batch {batch_idx+1} of epoch {epoch+1} at {timestamp}")

            if c.DISPLAY_VALIDATION_IMGS: display_validation_images(img_anaglyph, img_reversed, generated_reversed)

    model.train()

def calculate_losses(model, validation_dl, device, loss_fns):
    """ Calculate the losses for the validation dataset."""
    model.eval()
    val_losses = {loss_name: 0 for loss_name in loss_fns}

    with torch.no_grad():
        for batch in validation_dl:
            img_anaglyph = batch['a'].to(device)
            img_reversed = batch['r'].to(device)
            generated_reversed = model(img_anaglyph)

            for loss_name, loss_fn in loss_fns.items():
                loss = loss_fn(generated_reversed, img_reversed)
                val_losses[loss_name] += loss.item()

    avg_val_losses = {loss_name: val_loss / len(validation_dl) for loss_name, val_loss in val_losses.items()}
    model.train()
    return avg_val_losses

def train_unet(model, train_dl, val_dl, device, timestamp, tqdm, writer=None):
    """ Train the UNet model."""

    writer_passed = True if writer is not None else False
    if not writer_passed:
        writer = SummaryWriter(log_dir=os.path.join(c.LOGS_PATH, f"{timestamp}"))

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=c.ADAM_LR)
    loss_fns = {
        'mse': ImageLoss(loss_type='mse', device=device),
        'mae': ImageLoss(loss_type='mae', device=device),
        'ssim': ImageLoss(loss_type='ssim', device=device),
        'psnr': ImageLoss(loss_type='psnr', device=device)
    }
    loss_names = list(loss_fns.keys())
    losses_csv_path = os.path.join(c.RESULTS_PATH, f"training_losses_unet_{timestamp}.csv")

    with open(losses_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["Epoch"] + [f"Train Loss ({name.upper()})" for name in loss_names] + [f"Validation Loss ({name.upper()})" for name in loss_names]
        csv_writer.writerow(header)

    for epoch in range(c.EPOCHS):
        model.train()
        train_losses = {loss_name: 0 for loss_name in loss_fns}

        loop = tqdm(train_dl, leave=True)
        for batch_idx, batch in enumerate(loop):
            anaglyph = batch['a'].to(device, non_blocking=True)
            reversed_image = batch['r'].to(device, non_blocking=True)

            outputs = model(anaglyph)
            losses = {loss_name: loss_fn(outputs, reversed_image) for loss_name, loss_fn in loss_fns.items()}
            optimize_loss = losses[c.OPTIMIZE_LOSS]

            optimizer.zero_grad()
            optimize_loss.backward()
            optimizer.step()

            for loss_name, loss in losses.items():
                train_losses[loss_name] += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{c.EPOCHS}]")
            loop.set_postfix_str(s=f"Loss ({c.OPTIMIZE_LOSS.upper()}): {optimize_loss.item():.4f}")

        avg_train_losses = {loss_name: train_loss / len(train_dl) for loss_name, train_loss in train_losses.items()}
        print(f"Epoch [{epoch+1}/{c.EPOCHS}], " + ", ".join([f"Loss ({loss_name.upper()}): {avg_loss:.4f}" for loss_name, avg_loss in avg_train_losses.items()]))

        avg_val_losses = calculate_losses(model, val_dl, device, loss_fns)
        print(f"Epoch [{epoch+1}/{c.EPOCHS}], " + ", ".join([f"Validation Loss ({loss_name.upper()}): {avg_loss:.4f}" for loss_name, avg_loss in avg_val_losses.items()]))

        with open(losses_csv_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch + 1] + [avg_train_losses[loss_name] for loss_name in loss_fns] + [avg_val_losses[loss_name] for loss_name in loss_fns])

        for loss_name in loss_names:
            writer.add_scalar(f"Loss/Train/{loss_name}", avg_train_losses[loss_name], epoch + 1)
            writer.add_scalar(f"Loss/Validation/{loss_name}", avg_val_losses[loss_name], epoch + 1)

        if ((epoch + 1) % c.NUM_STORE_EVERY == 0) or ((epoch + 1) == c.EPOCHS):
            if c.STORE_VALIDATION_IMGS or c.DISPLAY_VALIDATION_IMGS: store_validation_images(model=model, validation_dl=val_dl, device=device, epoch=epoch, timestamp=timestamp)
            checkpoint_path = os.path.join(c.MODEL_PATH, f"unet_checkpoint_{timestamp}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    if not writer_passed:
        writer.close()