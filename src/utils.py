import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import EnhancementDataset

# peak signal-to-noise ratio
def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

class BalancedEnhancementLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, enhanced, high_res):
        mse_loss = self.mse(enhanced, high_res)
        enhancement_loss = 1.0 / (psnr(enhanced, high_res) + 1e-8)
        return self.alpha * mse_loss + self.beta * enhancement_loss

# We may change it during training automatically
def adjust_loss_weights(epoch, total_epochs):
    alpha = 1.0 - (epoch / total_epochs) * 0.3
    beta = (epoch / total_epochs) * 0.3
    return alpha, beta

# def visualize_results(model, low_res_image, high_res_image, device='cpu'):
#     model.eval()
#     pass


def check_data_availability(train_dir, val_dir):
    train_low_res = os.path.join(train_dir, 'low_res')
    train_high_res = os.path.join(train_dir, 'high_res')
    val_low_res = os.path.join(val_dir, 'low_res')
    val_high_res = os.path.join(val_dir, 'high_res')
    
    directories = [train_low_res, train_high_res, val_low_res, val_high_res]
    
    for directory in directories:
        if not os.path.exists(directory):
            return False, directory
    
    return True, None

def load_model(model_path, model_class, model_params):
    model = model_class(**model_params)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model