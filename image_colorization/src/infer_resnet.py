## Update: 11th Jun, 2024
## this file is supposed to give you a general idea on how to
## use the pre-trained model for colorizing B&W images. This
## file still needs development.

import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from models import MainModel, build_res_unet
from utils import lab_to_rgb

if __name__ == '__main__':
    model = MainModel()
    # You first need to download the final_model_weights.pt file from my drive
    # using the command: gdown --id 1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV
    # You need to train the Dynamic UNet generator and save the weights in a file resnet_model_weights.pt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(
        torch.load(
            "resnet_model_weights.pt",
            map_location=device
        )
    )
    model = MainModel(net_G=net_G)
    model.load_state_dict(
        torch.load(
            "../final_model_weights.pt",
            map_location=device
        )
    )
    path = "Path to your black and white image"
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    plt.imshow(colorized)