import torch
from torchvision import transforms
from PIL import Image
from unet import UNet
import argparse

def load_model(model_path, device):
    model = UNet()  # Initialize the U-Net model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_height, img_width, device):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def save_image(tensor, output_path):
    image = tensor.squeeze(0).cpu()
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save(output_path)

def infer_unet(model, image_path, output_path, img_height, img_width, device):
    image = preprocess_image(image_path, img_height, img_width, device)
    with torch.no_grad():
        output = model(image)
    save_image(output, output_path)
    print(f"Reversed anaglyph saved at {output_path}")

def create_stereo_pairs(anaglyph_path, reversed_path, stereo_output_path):
    anaglyph = Image.open(anaglyph_path).convert('RGB')
    reversed_anaglyph = Image.open(reversed_path).convert('RGB')

    anaglyph_r, anaglyph_g, anaglyph_b = anaglyph.split()
    reversed_r, reversed_g, reversed_b = reversed_anaglyph.split()

    left_image = Image.merge('RGB', (anaglyph_r, reversed_g, reversed_b))
    right_image = Image.merge('RGB', (reversed_r, anaglyph_g, anaglyph_b))

    stereo_pair = Image.new('RGB', (anaglyph.width * 2, anaglyph.height))
    stereo_pair.paste(left_image, (0, 0))
    stereo_pair.paste(right_image, (anaglyph.width, 0))

    stereo_pair.save(stereo_output_path)
    print(f"Stereo pair saved at {stereo_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process anaglyph images with U-Net model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the U-Net model checkpoint.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input anaglyph image.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the reversed anaglyph image.')
    parser.add_argument('--stereo_output_path', type=str, required=True, help='Path to save the stereo pair image.')
    parser.add_argument('--img_height', type=int, default=512, help='Height of the image.')
    parser.add_argument('--img_width', type=int, default=512, help='Width of the image.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, device)
    infer_unet(model, args.image_path, args.output_path, args.img_height, args.img_width, device)
    create_stereo_pairs(args.image_path, args.output_path, args.stereo_output_path)