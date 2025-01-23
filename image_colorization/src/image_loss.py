import torch.nn as nn
from piqa import SSIM, PSNR


class ImageLoss(nn.Module):
    def __init__(self, loss_type='mse', device='cpu'):
        super().__init__()
        self.device = device
        self.loss_type = loss_type.lower()

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == 'ssim':
            self.loss_fn = self.ssim_loss
        elif self.loss_type == 'psnr':
            self.loss_fn = self.psnr_loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def ssim_loss(self, img1, img2):
        criterion = SSIM().cuda() if self.device == 'cuda' else SSIM()
        return 1 - criterion(img1, img2)

    def psnr_loss(self, img1, img2):
        criterion = PSNR().cuda() if self.device == 'cuda' else PSNR()
        psnr = criterion(img1, img2)
        return -psnr

    def forward(self, preds, targets):
        return self.loss_fn(preds, targets)