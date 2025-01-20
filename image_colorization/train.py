import torch
from torch.optim import Adam
from image_colorization.dataloader_anaglyph import make_dataloaders
from image_colorization.discriminator import Discriminator
from image_colorization.generator import Generator
import config as c
from tqdm import tqdm

# Initialize hyperparameters
num_epochs = c.EPOCHS
adversarial_loss = torch.nn.BCELoss()
reconstruction_loss = torch.nn.L1Loss()
adam_lr = c.ADAM_LR
adam_beta1 = c.ADAM_BETA1
real_label = 1.0
fake_label = 0.0

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Dataloader
dataloader = make_dataloaders(path_anaglyph=c.TRAIN_ANAGLYPH_FILE, path_left=c.TRAIN_LEFT_FILE, path_right=c.TRAIN_RIGHT_FILE)

def train_gan(generator, discriminator, dataloader, num_epochs, device, lr, beta1):

    # Move models to device
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    optimizer_G = Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        g_loss_epoch = 0.0  # Accumulate generator loss for the epoch
        d_loss_epoch = 0.0  # Accumulate discriminator loss for the epoch

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Extract images from batch and preprocess
            img_anaglyph = batch['a'].to(device)  # Preprocess anaglyph image
            img_left =batch['l'].to(device)      # Preprocess left stereo image
            img_right = batch['r'].to(device)     # Preprocess right stereo image

            # ======== Train Discriminator ========
            optimizer_D.zero_grad()

            # Train with real stereo images
            output_real = discriminator(img_left, img_right).view(-1)
            label_real = torch.full_like(output_real, real_label, device=device)
            loss_real = adversarial_loss(output_real, label_real)

            # Train with fake stereo images
            fake_left, fake_right = generator(img_anaglyph)
            output_fake = discriminator(fake_left.detach(), fake_right.detach()).view(-1)
            label_fake = torch.full_like(output_fake, fake_label, device=device)
            loss_fake = adversarial_loss(output_fake, label_fake)

            # Combine losses
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # ======== Train Generator ========
            optimizer_G.zero_grad()

            # Generator tries to fool the discriminator
            output_fake = discriminator(fake_left, fake_right).view(-1)
            label_fake_for_gen = torch.full_like(output_fake, real_label, device=device)  # Trick discriminator
            loss_G = adversarial_loss(output_fake, label_fake_for_gen)

            loss_G.backward()
            optimizer_G.step()

            # Update epoch loss
            d_loss_epoch += loss_D.item()
            g_loss_epoch += loss_G.item()

        # Log epoch losses
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Discriminator Loss: {d_loss_epoch/len(dataloader):.4f}, "
              f"Generator Loss: {g_loss_epoch/len(dataloader):.4f}")


if __name__ == "__main__":
    train_gan(generator, discriminator, dataloader, num_epochs, device, adam_lr, adam_beta1)