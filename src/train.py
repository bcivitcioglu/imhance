import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset import EnhancementDataset
from utils import BalancedEnhancementLoss, adjust_loss_weights
from model.image_enhance_transformer import ImageEnhanceTransformer

def train_model(model, train_loader, val_loader, num_epochs=50, device='cpu'):
    model = model.to(device)
    criterion = BalancedEnhancementLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        alpha, beta = adjust_loss_weights(epoch, num_epochs)
        criterion.alpha, criterion.beta = alpha, beta

        for batch_idx, (low_res, high_res) in tqdm(enumerate(train_loader)):
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            optimizer.zero_grad()
            enhanced = model(low_res)
            loss = criterion(low_res, high_res, enhanced)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = validate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for low_res, high_res in val_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            enhanced = model(low_res)
            loss = criterion(low_res, high_res, enhanced)
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 4
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_dataset = EnhancementDataset('data/train/low_res', 'data/train/high_res')
    val_dataset = EnhancementDataset('data/val/low_res', 'data/val/high_res')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = ImageEnhanceTransformer(
        image_size=(256, 256),  # Adjust based on your image size
        patch_size=16,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        output_size=(256, 256)  # Adjust based on your desired output size
    )
    print("Beginning the training...")
    # Train
    trained_model = train_model(model, train_loader, val_loader, num_epochs, device)

    # Save the model
    torch.save(trained_model.state_dict(), 'trained_model.pth')