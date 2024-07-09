import torch
from torch import nn
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.w_q = nn.Linear(embed_dim,embed_dim,bias=False) # This is a linear transformation, therefore it is a matrix
        self.w_k = nn.Linear(embed_dim,embed_dim,bias=False)
        self.w_v = nn.Linear(embed_dim,embed_dim,bias=False)

        # This is a bit tricky, we need an output projection
        # Output projection is going to map the concatanated 
        # output of all heads (shape: (batch_size,num_patches,num_heads))
        # into the same dimension, but we do a projection so that 
        # each heads output can also go through learnable parameters
        # for better results

        self.w_out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        (batch_size, num_patches, embed_dim) = x.shape # It is important and vital that the dimensions of the input are correct

        # Now we create q,k,v
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
                     
        # Reshape to shape: (batch_size, num_patches, num_heads, head_dim), we divide embed_dim into num_heads and head_dim
        q = q.reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # Then we perform a permute
        k = k.reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # The permute is important
        # We had, after reshaping,  shape: (batch_size, num_patches, num_heads, head_dim) 
        # After permute we have shape: (batch_size, num_heads, num_patches, head_dim) 
        # The aim is to separate the heads, by making (num_patches, head_dim) full heads themselves
        # A little tricky but makes sense

        attention = torch.matmul(q,k.transpose(-2,-1)) * self.head_dim ** -0.5 # When we check the dimensions here
        # q has shape: (batch_size, num_heads, num_patches, head_dim) 
        # k has shape: (batch_size, num_heads, num_patches, head_dim) 
        # batch_size and num_heads are not the matrix operation, they are parallelization
        # Our matrix operation is in the last two dimensions
        # So when we switch them with k.transpose(-2,-1) then we can make the matrix operation

        # attention has shape: (batch_size, num_heads, num_patches, num_patches)
    
        attention_prob = torch.softmax(attention,dim=-1)
        # attention_prob has shape: (batch_size, num_heads, num_patches, num_patches)
        # softmax does not change the dimensions but only values

        attention_out = torch.matmul(attention_prob, v) 
        # v has shape: (batch_size, num_heads, num_patches, head_dim)
        # The matrix multiplication is between:
        # attention_prob, shape: (batch_size, num_heads, num_patches, num_patches)
        # v, shape: (batch_size, num_heads, num_patches, head_dim)

        # attention_out has shape: (batch_size, num_heads, num_patches, head_dim)
        # This gives us the weighted sum of values for each query patch

        attention_out = attention_out.permute(0, 2, 1, 3).reshape(batch_size, num_patches, embed_dim) # Reshaped to the input shape, just like before
        output = self.w_out(attention_out) # Output projection applied, keeping the dimension but adding context between heads results

        return output