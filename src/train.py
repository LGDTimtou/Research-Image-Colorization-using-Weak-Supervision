import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import SIGGRAPHGenerator
from skimage.color import rgb2lab
from tqdm import trange, tqdm
import numpy as np
from sampling_options import SamplingOption
from util import get_surrounding_coords
from datetime import datetime

debug = True

DATA_FOLDER = 'data/' if debug else '/data/gent/465/vsc46581/ml'


def epoch_loop(model: SIGGRAPHGenerator, device, dataloader, criterion, user_input_distribution, user_input_n, user_input_p, optimizer=None, training=True):
    epoch_loss = 0.0
    for inputs, _ in tqdm(dataloader):
        inputs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        images_lab = []
        inputs_ab = []
        inputs_mask = []
        for img in inputs:
            img_lab = rgb2lab(img)

            input_ab, input_mask = simulate_user_inputs(img_lab[:, :, 1:], user_input_distribution, user_input_n, user_input_p)
            images_lab.append(img_lab)
            inputs_ab.append(input_ab)
            inputs_mask.append(input_mask)

        img_lab = np.stack(images_lab, axis=0)
        input_ab = np.stack(inputs_ab, axis=0)
        input_mask = np.stack(inputs_mask, axis=0)


        img_l = torch.from_numpy(img_lab[:, :, :, :1]).permute(0, 3, 1, 2).float().to(device)
        img_ab = torch.from_numpy(img_lab[:, :, :, 1:]).permute(0, 3, 1, 2).float().to(device)


        input_ab = torch.from_numpy(input_ab).permute(0, 3, 1, 2).to(device)
        input_mask = torch.from_numpy(input_mask).permute(0, 3, 1, 2).to(device)

        if training:
            optimizer.zero_grad()

        output_ab = model(img_l - 50, input_ab, input_mask)
        loss = criterion(output_ab, img_ab)

        if training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def simulate_user_inputs(ab_image, distribution, n, p):
    points = distribution.call_function(n, 64)
    input_ab = np.zeros_like(ab_image)

    for x, y in points:
        x1, y1, x2, y2 = get_surrounding_coords(x, y, p, 64)
        region = ab_image[y1:y2, x1:x2]
        mean_color = region.mean(axis=(0, 1))

        input_ab[y1:y2, x1:x2, :] = mean_color  

    input_mask = np.max(input_ab, axis=2) > 0
    input_mask = input_mask[:, :, np.newaxis]
    input_mask = input_mask.astype(np.float32)

    return input_ab, input_mask


def train(model: SIGGRAPHGenerator, dataloader, criterion, optimizer, device, user_input_distribution, user_input_n, user_input_p):
    model.train()
    return epoch_loop(model, device, dataloader, criterion, user_input_distribution, user_input_n, user_input_p, optimizer)
    

def validate(model: SIGGRAPHGenerator, dataloader, criterion, device, user_input_distribution, user_input_n, user_input_p):
    model.eval()
    return epoch_loop(model, device, dataloader, criterion, user_input_distribution, user_input_n, user_input_p, training=False)

# Main script
if __name__ == "__main__":
    # Configurations
    batch_size = 32 
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset transformation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=transform)

    # Split into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Define DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    model = SIGGRAPHGenerator().to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Setting user input simulation variables
    distribution = SamplingOption.GAUSSIAN
    n = 20
    p = 2

    output_file = f"output/training_{str(datetime.now())}".replace(" ", "_")
    output_string = f"Training on {device}\nbatch size: {batch_size}\n#epochs: {num_epochs}\nlearning rate: {learning_rate}\noptimizer: {str(optimizer)}\nloss function: {str(criterion)}\nuser input simulation variables:\n - distribution: {str(distribution)}\n - n: {n}\n - p: {p}"
    print(output_string)

    with open(output_file + ".txt", 'a+') as f:
        f.write(output_string)

    # Training loop
    best_val_loss = float('inf')
    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, distribution, n, p)
        val_loss = validate(model, val_loader, criterion, device, distribution, n, p)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{output_file}.pth")
            print(f"Validation loss improved. Saving model at epoch {epoch + 1}")

    print("Training complete.")
    
    
