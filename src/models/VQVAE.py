import torch
import torch.nn as nn
import torch.nn.functional as F
# from performer_pytorch import PerformerLM
import math
from PIL import Image
import numpy as np

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]


        distances = (
            torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.codebook.weight.t())
        )


        encoding_indices = torch.argmin(distances, dim=1)


        z_q = self.codebook(encoding_indices).view(B, H, W, C)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        # Loss
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_e.detach(), z_q)

        return z_q, codebook_loss + 0.25 * commitment_loss, encoding_indices

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        groups = min(8, out_channels // 4) if out_channels >= 4 else 1
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        if self.residual:
            return nn.SiLU()(self.skip(x) + self.double_conv(x))
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=in_channels//2,
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv = DoubleConv(in_channels//2 + skip_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.up(x)
        if x_skip.shape[1] != x.shape[1]:
            x_skip = F.interpolate(x_skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x_skip], dim=1)
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels) 
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H * W == self.size * self.size, f"Expected size {self.size}x{self.size}, got {H}x{W}"
        

        x = x.view(B, C, -1).permute(0, 2, 1)
        

        x_ln = self.ln(x)  
        
 
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        

        attention_value = self.ff_self(attention_value) + attention_value
        

        return attention_value.permute(0, 2, 1).view(B, C, H, W)


class VQVAE_UNet(nn.Module):
    def __init__(self, c_in=4, c_out=4, img_size=128, num_embeddings=512, embedding_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        
        # Encoder
        self.inc0 = DoubleConv(c_in, 128)
        self.inc = DoubleConv(128, 32)
        self.inc2 = DoubleConv(32, 32, residual=True)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)  # img_size/4 = 32
        self.down3 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)  # img_size/8 = 16
        self.down4 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)   # img_size/16 = 8
        
        # Quantizer
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        
        # Decoder
        self.bot1 = DoubleConv(256, 512)
        self.bot3 = DoubleConv(512, 256)
        self.sa4 = SelfAttention(256, 8)
        self.up1 = Up(256, 128, 256) 
        self.sa5 = SelfAttention(128, 16)
        self.up2 = Up(128, 64, 128) 
        self.sa6 = SelfAttention(64, 32)
        self.up3 = Up(64, 32, 64)   
        self.up4 = Up(32, 32, 32)    
        self.outc = nn.Conv2d(32, c_out, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc0(x)      
        x1 = self.inc(x1)         
        x1 = self.inc2(x1)       
        x2 = self.down1(x1)        
        x3 = self.down2(x2)        
        x3 = self.sa1(x3)         
        x4 = self.down3(x3)     
        x4 = self.sa2(x4)          
        x5 = self.down4(x4)     
        x5 = self.sa3(x5)     
        
        # Quantization
        z_q, vq_loss, encoding_indices = self.quantizer(x5) 
        
        # Decoder
        x5 = self.bot1(z_q)   
        x5 = self.bot3(x5)       
        x = self.sa4(x5)         
        x = self.up1(x, x4)       
        x = self.sa5(x)           
        x = self.up2(x, x3)        
        x = self.sa6(x)         
        x = self.up3(x, x2)       
        x = self.up4(x, x1)       
        x_recon = self.outc(x)     
        
        return x_recon, vq_loss, encoding_indices


class TransformerModel(nn.Module):
    def __init__(self, num_tokens=512, dim=256, depth=6, img_size=128):
        super().__init__()
        latent_size = img_size // 16
        self.max_seq_len = latent_size * latent_size
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=8,
            dim_feedforward=4*dim,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        self.head = nn.Linear(dim, num_tokens)
    
    def forward(self, x):
        # x: [B, seq_len]
        seq_len = x.size(1) 
        x = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        return self.head(self.transformer(x))
