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

    def forward(self, low_res, high_res, enhanced):
        mse_loss = self.mse(enhanced, high_res)
        enhancement_loss = 1.0 / (psnr(enhanced, low_res) + 1e-8)
        return self.alpha * mse_loss + self.beta * enhancement_loss

# We may change it during training automatically
def adjust_loss_weights(epoch, total_epochs):
    alpha = 1.0 - (epoch / total_epochs) * 0.3
    beta = (epoch / total_epochs) * 0.3
    return alpha, beta

# def visualize_results(model, low_res_image, high_res_image, device='cpu'):
#     model.eval()
#     pass