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

def preprocess_image(image, img_height, img_width, device):
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def infer_nn_model(model, image, img_height, img_width, device):
    image = preprocess_image(image, img_height, img_width, device)
    with torch.no_grad():
        output = model(image)
    return output

def create_stereo_pairs(anaglyph, reversed_image):
    reversed_anaglyph = transforms.ToPILImage()(reversed_image.squeeze(0).cpu())

    anaglyph_r, anaglyph_g, anaglyph_b = anaglyph.split()
    reversed_r, reversed_g, reversed_b = reversed_anaglyph.split()

    left_image = Image.merge('RGB', (anaglyph_r, reversed_g, reversed_b))
    right_image = Image.merge('RGB', (reversed_r, anaglyph_g, anaglyph_b))

    stereo_pair = Image.new('RGB', (anaglyph.width * 2, anaglyph.height))
    stereo_pair.paste(left_image, (0, 0))
    stereo_pair.paste(right_image, (anaglyph.width, 0))

    return stereo_pair

def display_3d_goggles():
    output = """
                                      _.....__
                             (.--...._`'--._
                   _,...----''''`-.._ `-..__`.._
          __.--'-;..-------'''''`._')      `--.-.__
        '-------------------------------------------'
        \ '----------------  ,-.  .-------------'. |
         \`.              ,','  \ \             ,' /
          \ \             / /   `.`.          ,' ,'
          `. `.__________/,'     `.' .......-' ,'
            `............-'        "---------''
            
            ______     _____ _                       
            | ___ \   /  ___| |                      
            | |_/ /___\ `--.| |_ ___ _ __ ___  ___   
            |    // _ \`--. \ __/ _ \ '__/ _ \/ _ \  
            | |\ \  __/\__/ / ||  __/ | |  __/ (_) | 
            \_| \_\___\____/ \__\___|_|  \___|\___/  
                                                     
        --- Recreate Stereo Images from its Anaglyph! ---
    """

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process anaglyph images with U-Net model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the U-Net model checkpoint.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input anaglyph image.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the reversed anaglyph image.')
    parser.add_argument('--stereo_output_path', type=str, required=True, help='Path to save the stereo pair image.')
    parser.add_argument('--model_input_height', type=int, default=256, help='Height of the input of the model.')
    parser.add_argument('--model_input_width', type=int, default=256, help='Width of the input of the model.')
    parser.add_argument('--output_height', type=int, default=256, help='Height of the output reversed anaglyph image.')
    parser.add_argument('--output_width', type=int, default=256, help='Width of the output reversed anaglyph image.')

    args = parser.parse_args()

    display_3d_goggles()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, device)

    anaglyph_image = Image.open(args.image_path).convert('RGB')
    reversed_image = infer_nn_model(model, anaglyph_image, args.model_input_height, args.model_input_width, device)

    reversed_anaglyph = transforms.ToPILImage()(reversed_image.squeeze(0).cpu())
    reversed_anaglyph = reversed_anaglyph.resize((args.output_width, args.output_height))
    reversed_anaglyph.save(args.output_path)
    print(f"Reversed anaglyph saved at {args.output_path}")

    stereo_pair = create_stereo_pairs(anaglyph_image, reversed_image)
    stereo_pair.save(args.stereo_output_path)
    print(f"Stereo pair saved at {args.stereo_output_path}")