import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from glob import glob
from src.dataset import EnhancementDataset
from src.model.image_enhance_transformer import ImageEnhanceTransformer
from src.train import train_model
from src.utils import check_data_availability
from src.test_utils import test_and_visualize_model

import config

import torch
from torch.utils.data import DataLoader


def main():
    # Data directories
    train_dir = 'data/train'
    val_dir = 'data/val'
    test_dir = 'data/test'
    model_path = glob("*.pth")
    # Check data availability
    data_available, missing_dir = check_data_availability(train_dir, val_dir)

    if data_available:
        print("All required data directories are available.")
    else:
        print(f"Error: Data directory not found: {missing_dir}")

    # Check for existing model
    if model_path.count == 1:
        print(f"Trained model found at {model_path}")
        print("Enter path to an image to test the model:")
        test_and_visualize_model(model_path, test_dir)
    else:
        print("No trained model found.")
        # Here could add logic to ask if the user wants to train a new model

if __name__ == "__main__":
    main()