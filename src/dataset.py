from torch.utils.data import Dataset
import os 
from PIL import Image
import torchvision.transforms as transforms
from glob import glob

class EnhancementDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None,low_res_size=(256, 256), high_res_size=(1024, 1024)):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir

        self.low_res_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(low_res_size)
        ])
        self.high_res_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(high_res_size)
        ])

        # Find all high-res images
        high_res_files = glob(os.path.join(high_res_dir, '*.[pj][np][g]'))
        
        # Create pairs of low-res and high-res files
        self.image_pairs = []
        for high_res_file in high_res_files:
            base_name = os.path.basename(high_res_file)
            name, ext = os.path.splitext(base_name)
            low_res_file = os.path.join(low_res_dir, f"{name}x2{ext}")
            if os.path.exists(low_res_file):
                self.image_pairs.append((low_res_file, high_res_file))

        print(f"Found {len(self.image_pairs)} image pairs")

    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, index):
        low_res_path, high_res_path = self.image_pairs[index]
        
        try:
            low_res_img = Image.open(low_res_path).convert('RGB')
            high_res_img = Image.open(high_res_path).convert('RGB')
        except IOError as e:
            print(f"Error opening image files: {e}")
            # Skip to the next item if possible
            if index + 1 < len(self):
                return self.__getitem__(index + 1)
            else:
                raise IOError(f"No valid image pair found for index {index}")
        
        # Apply transformations
        low_res_tensor = self.low_res_transform(low_res_img)
        high_res_tensor = self.high_res_transform(high_res_img)
        
        return low_res_tensor, high_res_tensor