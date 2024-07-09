from torch import nn 
from model.transformer import TransformerEncoder
from model.patch_embedding import PatchEmbedding

class ImageEnhanceTransformer(nn.Module):
    def __init__(self, image_size : tuple, patch_size, embed_dim, num_layers, num_heads, output_size:tuple, in_channels = 3):
        super().__init__()

        self.img_height, self.img_width = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.out_height, self.out_width = output_size
        self.in_channels = in_channels

        self.patch_embed = PatchEmbedding((self.img_height,self.img_width),self.patch_size,self.embed_dim,self.in_channels)

        assert self.img_height == self.patch_embed.effective_h, "Image height doesn't match effective height"
        assert self.img_width == self.patch_embed.effective_w, "Image width doesn't match effective width"

        self.encoder = TransformerEncoder(self.num_layers,self.embed_dim,self.num_heads)

        self.num_patches_h = self.patch_embed.num_patches_h
        self.num_patches_w = self.patch_embed.num_patches_w
        self.num_patches = self.patch_embed.num_patches



        # Linear projection for reconstruction
        self.to_patch = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * self.in_channels)

        # a layer for resizing if output_size is different from input_size
        self.need_resize = (self.out_height, self.out_width) != (self.img_height, self.img_width)
        if self.need_resize:
            self.resize = nn.Upsample(size=(self.out_height, self.out_width), mode='bilinear', align_corners=False)

    def forward(self,x):
        # Input dimension: (batch_size, in_channels, img_height, img_width)
        x = self.patch_embed(x)
        # After patch embedding: (batch_size, num_patches, embed_dim)
        x = self.encoder(x)
        # After encoder: (batch_size, num_patches, embed_dim)
        x = self.to_patch(x)
        # After projection: (batch_size, num_patches, patch_size * patch_size * in_channels)

        # Now we need to reshape the output and make sure we are returning back 
        # the desired output image sizes
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_patches_h, self.num_patches_w, 
                   self.patch_size, self.patch_size, self.in_channels)
        # After first reshape: (batch_size, num_patches_h, num_patches_w, patch_size, patch_size, in_channels)
        # We separate all variables to regroup
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous() # contigous makes a copy of the original tensor in the memory, instead of just changing meta information
        # After permute: (batch_size, in_channels, num_patches_h, patch_size, num_patches_w, patch_size)
        # We put num_patches_h and patch_size adjacently, similar to num_patches_w
        x = x.view(batch_size, self.in_channels, self.img_height, self.img_width) # like reshape but contigous
        # After view: (batch_size, self.in_channels, self.img_height, self.img_width)
        # Then we get the img_heigh and img_width back
        # Resize if necessary
        if self.need_resize:
            x = self.resize(x)
        
        return x
