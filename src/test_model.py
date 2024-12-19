import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from util import get_surrounding_coords
import numpy as np
from model import SIGGRAPHGenerator
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from sampling_options import SamplingOption
import random
import json

# Load CIFAR10 dataset with resized images to 64x64
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load pre-trained models
model_paths = ["models/model_GAUSSIAN.pth", "models/model_POISSON.pth", "models/model_RANDOM.pth"]
models = []
for path in model_paths:
    model = SIGGRAPHGenerator()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    models.append(model)

def calculate_metrics(original_l, generated_ab, original_ab):
    original_l = original_l.permute(1, 2, 0).cpu().numpy()
    generated_ab = generated_ab.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    original_ab = original_ab.permute(1, 2, 0).cpu().numpy()

    original_lab = np.concatenate([original_l, original_ab], axis=-1)
    generated_lab = np.concatenate([original_l, generated_ab], axis=-1)

    original_rgb = (lab2rgb(original_lab) * 255).astype(np.uint8)
    generated_rgb = (lab2rgb(generated_lab) * 255).astype(np.uint8)

    ssim_value = ssim(original_rgb, generated_rgb, multichannel=True, data_range=255, win_size=3)
    psnr_value = psnr(original_rgb, generated_rgb, data_range=255)

    return ssim_value, psnr_value

def simulate_user_inputs(ab_image, user_input):
    points = user_input['distribution'].call_function(user_input['n'], 64)
    input_ab = np.zeros_like(ab_image)

    for x, y in points:
        x1, y1, x2, y2 = get_surrounding_coords(x, y, user_input['p'], 64)
        region = ab_image[y1:y2, x1:x2]
        mean_color = region.mean(axis=(0, 1))

        input_ab[y1:y2, x1:x2, :] = mean_color  

    input_mask = np.max(input_ab, axis=2) > 0
    input_mask = input_mask[:, :, np.newaxis]
    input_mask = input_mask.astype(np.float32)

    return input_ab, input_mask


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, model in enumerate(models):
        model.to(device)
        model_results = []
        total_ssim, total_psnr = 0, 0
        for j, (inputs, _) in tqdm(enumerate(dataloader)):
            inputs = inputs.permute(0, 2, 3, 1).cpu().numpy()

            img = inputs[0]

            img_lab = rgb2lab(img)

            user_input = {"distribution": SamplingOption.GAUSSIAN, "n": random.choice([3, 5, 8, 10, 15, 20]), "p": 3}

            
            input_ab, input_mask = simulate_user_inputs(img_lab[:, :, 1:], user_input)
            input_ab = torch.from_numpy(input_ab).permute(2, 0, 1).to(device)
            input_mask = torch.from_numpy(input_mask).permute(2, 0, 1).to(device)

            img_l = torch.from_numpy(img_lab[:, :, :1]).permute(2, 0, 1).float().to(device)
            img_ab = torch.from_numpy(img_lab[:, :, 1:]).permute(2, 0, 1).float().to(device)

            output_ab = model(img_l - 50, input_ab, input_mask)

            ssim_value, psnr_value = calculate_metrics(img_l, output_ab, img_ab)
            model_results.append({
                "image_idx": j,
                "ssim": ssim_value,
                "psnr": psnr_value,
                "user_input": {
                    "distribution": str(user_input["distribution"]),  # Convert SamplingOption to string
                    "n": user_input["n"],
                    "p": user_input["p"],
                },
            })
            total_ssim += ssim_value
            total_psnr += psnr_value


        avg_ssim = total_ssim / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        with open(f"output/results_model_{i}.json", "w") as f:
            json.dump({"results": model_results, "avg_ssim": avg_ssim, "avg_psnr": avg_psnr}, f, indent=2)






if __name__ == "__main__":
    test()