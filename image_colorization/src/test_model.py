import os
import torch
import csv
from tqdm import tqdm
from src.image_loss import ImageLoss

def set_global_config(config_module):
    global c
    c = config_module

def calculate_test_losses(model, test_dl, device, loss_fns, csv_writer):
    model.eval()
    batch_num = 1

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Testing", unit="batch"):
            img_anaglyph = batch['a'].to(device)
            img_reversed = batch['r'].to(device)
            generated_reversed = model(img_anaglyph)

            batch_losses = []
            for loss_name, loss_fn in loss_fns.items():
                loss = loss_fn(generated_reversed, img_reversed)
                batch_losses.append(loss.item())

            csv_writer.writerow([batch_num] + batch_losses)
            batch_num += 1

    model.train()

def test_model(model, test_dl, device, timestamp):
    model = model.to(device)
    loss_fns = {
        'mse': ImageLoss(loss_type='mse', device=device),
        'mae': ImageLoss(loss_type='mae', device=device),
        'ssim': ImageLoss(loss_type='ssim', device=device),
        'psnr': ImageLoss(loss_type='psnr', device=device)
    }
    loss_names = list(loss_fns.keys())
    losses_csv_path = os.path.join(c.RESULTS_PATH, f"testing_losses_unet_{timestamp}.csv")

    with open(losses_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["Batch"] + [f"Test Loss ({name.upper()})" for name in loss_names]
        csv_writer.writerow(header)
        print(f"Header written to {losses_csv_path}")

        calculate_test_losses(model, test_dl, device, loss_fns, csv_writer)
        print(f"Test results written to {losses_csv_path}")