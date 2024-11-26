import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms
from model import SIGGRAPHGenerator
from skimage.color import rgb2lab
from tqdm import trange, tqdm
import numpy
from sampling_options import SamplingOption
from util import get_surrounding_coords
from datetime import datetime
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)

debug = True

DATA_FOLDER = 'data/' if debug else '/data/gent/465/vsc46581/ml'


def epoch_loop(model: SIGGRAPHGenerator, device, dataloader, criterion, user_input, optimizer=None, option=0):
    epoch_loss = 0.0
    total_ciede = 0.0
    for inputs, _ in tqdm(dataloader):
        inputs = inputs.permute(0, 2, 3, 1).cpu().numpy()

        images_lab = []
        inputs_ab = []
        inputs_mask = []
        for img in inputs:
            img_lab = rgb2lab(img)

            input_ab, input_mask = simulate_user_inputs(img_lab[:, :, 1:], user_input)
            images_lab.append(img_lab)
            inputs_ab.append(input_ab)
            inputs_mask.append(input_mask)

        img_lab = numpy.stack(images_lab, axis=0)
        input_ab = numpy.stack(inputs_ab, axis=0)
        input_mask = numpy.stack(inputs_mask, axis=0)


        img_l = torch.from_numpy(img_lab[:, :, :, :1]).permute(0, 3, 1, 2).float().to(device)
        img_ab = torch.from_numpy(img_lab[:, :, :, 1:]).permute(0, 3, 1, 2).float().to(device)


        input_ab = torch.from_numpy(input_ab).permute(0, 3, 1, 2).to(device)
        input_mask = torch.from_numpy(input_mask).permute(0, 3, 1, 2).to(device)

        if option == 0:
            optimizer.zero_grad()

        output_ab = model(img_l - 50, input_ab, input_mask)
        loss = criterion(output_ab, img_ab)

        if option == 1:
            total_ciede += calculate_ciede2000(img_l, img_ab, output_ab)

        if option == 0:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return total_ciede / len(dataloader) if option == 1 else epoch_loss / len(dataloader)


def simulate_user_inputs(ab_image, user_input):
    points = user_input['distribution'].call_function(user_input['n'], 64)
    input_ab = numpy.zeros_like(ab_image)

    for x, y in points:
        x1, y1, x2, y2 = get_surrounding_coords(x, y, user_input['p'], 64)
        region = ab_image[y1:y2, x1:x2]
        mean_color = region.mean(axis=(0, 1))

        input_ab[y1:y2, x1:x2, :] = mean_color  

    input_mask = numpy.max(input_ab, axis=2) > 0
    input_mask = input_mask[:, :, numpy.newaxis]
    input_mask = input_mask.astype(numpy.float32)

    return input_ab, input_mask

def calculate_ciede2000(img_l, img1_ab, img2_ab):
    batch_size, h, w, _ = img1_ab.shape
    total_diff = 0.0

    for i in range(batch_size):
        for j in range(h):
            for k in range(w):
                color1 = LabColor(lab_l=50.0, lab_a=img1_ab[i, j, k, 0], lab_b=img1_ab[i, j, k, 1])
                color2 = LabColor(lab_l=50.0, lab_a=img2_ab[i, j, k, 0], lab_b=img2_ab[i, j, k, 1])
                total_diff += delta_e_cie2000(color1, color2)

    return total_diff / (h * w)


def train(model: SIGGRAPHGenerator, device, dataloader, criterion, user_input, optimizer):
    model.train()
    return epoch_loop(model, device, dataloader, criterion, user_input, optimizer)
    

def validate(model: SIGGRAPHGenerator, device, dataloader, criterion, user_input):
    model.eval()
    return epoch_loop(model, device, dataloader, criterion, user_input, option=-1)

def test(model: SIGGRAPHGenerator, device, dataloader, criterion, output_file):
    model.eval()
    distributions = [option for option in SamplingOption]
    ns = [1, 3, 5, 10, 15, 20, 30]
    ps = [1, 2, 3, 4, 5]

    for distribution in distributions:
        for n in ns:
            for p in ps:
                user_input_params = {
                    "distribution": distribution,
                    "n": n,
                    "p": p
                }
                ciede2000 = epoch_loop(model, device, dataloader, criterion, user_input_params, option=1)
                with open("output/" + output_file + ".txt", 'a+') as f:
                    f.write(f" - distribution: {distribution}\n - n: {n}\n - p: {p}\n - average ciede loss: {ciede2000}\n-------------------------------------------\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define parameters
    batch_size = 32 
    num_epochs = 20
    learning_rate = 0.001

    distribution = SamplingOption.GAUSSIAN
    n = 20
    p = 2

    user_input_params = {
        "distribution": distribution,
        "n": n,
        "p": p
    }

    output_file = f"training_{str(datetime.now())}".replace(" ", "_")
    output_string = f"Training on {device}\nbatch size: {batch_size}\n#epochs: {num_epochs}\nlearning rate: {learning_rate}\nuser input simulation variables:\n - distribution: {str(distribution)}\n - n: {n}\n - p: {p}"
    print(output_string)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    model = SIGGRAPHGenerator().to(device)
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    with open("output/" + output_file + ".txt", 'a+') as f:
        f.write(output_string + '\n\nTest results:')


    # Training
    train_dataset = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    for epoch in trange(num_epochs):
        train_loss = train(model, device, train_loader, criterion, user_input_params, optimizer)
        val_loss = validate(model, device, val_loader, criterion, user_input_params)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"models/{output_file}.pth")
            print(f"Validation loss improved. Saving model at epoch {epoch + 1}")

    print("Training complete!")
    # Testing
    print("Testing...")
    test_dataset_train = datasets.ImageFolder(root=DATA_FOLDER + 'imagenette2-160/train', transform=transform)
    test_dataset_val = datasets.ImageFolder(root=DATA_FOLDER + 'imagenette2-160/val', transform=transform)

    test_loader = DataLoader(ConcatDataset([test_dataset_train, test_dataset_val]), batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load(f"models/caffemodel.pth", map_location=device))
    test(model, device, test_loader, criterion, output_file)

    print("Testing complete!")

    
main()