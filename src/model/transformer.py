from torch import nn
from model.attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,ff_dim=2048):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.multi_head_att = MultiHeadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)

        # Feed-Forward Network
        # This is usually used as two layers in transformers
        # expand-then-contract type
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

        # We make layer normalization
        # This normalizes the values of the inputs across the features
        # it makes mean normalization, with learnable parameters, this is why we 
        # have two separate ones
        # 2 parameters: one is normalization * gamma: gamma gives different importance to different values
        # + beta: gives optimum bias when leant 
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # First sub-layer: Multi-Head Attention with residual connection and layer norm
        # what is residual connection? 
        # Adding the input, back to the output after the attention
        norm1 = self.norm1(x)
        attn_output = self.multi_head_att(norm1)
        x = x + attn_output  # Residual connection
        
        # Second sub-layer: Feed-Forward Network with residual connection and layer norm
        norm2 = self.norm2(x)
        ff_output = self.ff_network(norm2)
        x = x + ff_output  # Residual connection
        
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        encoder_list = [TransformerEncoderBlock(embed_dim=self.embed_dim, num_heads=self.num_heads) for _ in range(num_layers)]
        self.encoder = nn.ModuleList(encoder_list)

    def forward(self,x):
        for layer in self.encoder:
            x = layer(x)
        return x
    