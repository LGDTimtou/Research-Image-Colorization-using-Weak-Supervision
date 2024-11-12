import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import SIGGRAPHGenerator
from skimage.color import rgb2lab
from tqdm import trange, tqdm

def epoch_loop(model: SIGGRAPHGenerator, device, dataloader, criterion, optimizer=None, training=True):
    epoch_loss = 0.0
    for inputs, _ in tqdm(dataloader):
        img_lab = rgb2lab(inputs[0].cpu().numpy(), channel_axis=0)
        img_l = torch.from_numpy(img_lab[:1, :, :]).float().to(device)
        img_ab = torch.from_numpy(img_lab[1:, :, :]).float().to(device)

        input_ab = torch.zeros_like(img_ab).to(device)
        input_mask = torch.zeros_like(img_l).to(device)
        
        if training:
            optimizer.zero_grad()

        output_ab = model(img_l, input_ab, input_mask)
        loss = criterion(output_ab[0], img_ab)

        if training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train(model: SIGGRAPHGenerator, dataloader, criterion, optimizer, device):
    model.train()
    return epoch_loop(model, device, dataloader, criterion, optimizer)
    

def validate(model: SIGGRAPHGenerator, dataloader, criterion, device):
    model.eval()
    return epoch_loop(model, device, dataloader, criterion, training=False)

# Main script
if __name__ == "__main__":
    # Configurations
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset transformation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Define DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=2)

    # Initialize model, criterion, optimizer
    model = SIGGRAPHGenerator().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in trange(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save the model checkpoint if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Validation loss improved. Saving model at epoch {epoch + 1}")

    print("Training complete.")
    
    
