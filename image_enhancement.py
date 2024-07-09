import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
print("import success")

class EnhancementDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.image_files = os.listdir(low_res_dir)

    def __len__(self):
        return len(self.image_files) # Return number of images
    
    def __getitem__(self,index):
        # Get the filename of the image
        img_name = self.image_files[index]
        
        # Construct full paths for low-res and high-res images
        low_res_path = os.path.join(self.low_res_dir, img_name)
        high_res_path = os.path.join(self.high_res_dir, img_name)
        
        # Open both images
        low_res_img = Image.open(low_res_path).convert('RGB')
        high_res_img = Image.open(high_res_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            low_res_img = self.transform(low_res_img)
            high_res_img = self.transform(high_res_img)
        
        return low_res_img, high_res_img
    

class PatchEmbedding(nn.Module):
    def __init__(self, image_size : tuple, patch_size, embed_dim, in_channels = 3):
        super().__init__()
        self.img_height, self.img_width = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Calculate the number of patches 
        self.num_patches_h = self.img_height // self.patch_size
        self.num_patches_w = self.img_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Effective image size 
        self.effective_h = self.num_patches_h * self.patch_size
        self.effective_w = self.num_patches_w * self.patch_size
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches , embed_dim))

    def __call__(self, batch_imgs: torch.Tensor):
        assert batch_imgs.dim() == 4, "Expecting batch of images with the shape (BATCH, CHANNEL, HEIGHT, WIDTH)"
        in_batch_size, in_chn, in_h, in_w = batch_imgs.shape
        assert (in_h == self.effective_h and in_w == self.effective_w), f"Input image shape must be ({self.effective_h}, {self.effective_w}) but given ({in_h,in_w})"
        
        # Apply patch embedding
        patch_embedded_image = self.patch_embed(batch_imgs)  # shape: (batch_size, embed_dim, height/patch_size, width/patch_size)
        
        # Reshape to sequence of patches
        patch_sequence_image = patch_embedded_image.flatten(2).transpose(1, 2)  
        # shape: (batch_size, num_patches, embed_dim)
        # flatten(2) combines the two last dimensions of patch_embedded_image, returning shape: (batch_size, embed_dim, num_patches)
        # transpose switches the last and the middle axes, resulting in shape: (batch_size, num_patches, embed_dim)
        
        # Add positional embedding
        patch_sequence_image += self.pos_embed
        
        return patch_sequence_image