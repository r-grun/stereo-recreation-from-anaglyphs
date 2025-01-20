import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: (6, H, W)
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, H/2, W/2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: (512, H/16, W/16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Output: (1, H/16 - 3, W/16 - 3)
            nn.Sigmoid()  # Probability map
        )

    def forward(self, left_image, right_image):
        # Concatenate left and right images along the channel dimension
        x = torch.cat([left_image, right_image], dim=1)  # Shape: (6, H, W)
        return self.model(x)
