import numbers
# from basicsr.models.archs.UNet_arch import Network
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings


##########################################################################
## Utils

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

##########################################################################
## Attention
class MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(MHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_in = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.proj_in(x)
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.proj_out(out)
        out = out.view(b, c, h, w)
        return out


class R_MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super(R_MHSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.proj_in = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x, prev_feat):
        b,c,h,w = x.shape

        qkv = self.proj_in(x)
        q,k,v = qkv.chunk(3, dim=1)   

        v = v * prev_feat
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # compute attn
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.proj_out(out)
        out = out.view(b, c, h, w)
        return out

class R_Masked_R_MHSA(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        """
        Args:
            dim (int): number of channels
            num_heads (int): number of attention heads
            bias (bool): whether to use bias in the Conv2d layers
        """
        super(R_Masked_R_MHSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Learnable temperatures for main attn and mask-based attn
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Projection layers to obtain Q, K, V for main x
        self.proj_in = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        
        # Projection layers to obtain Q_m, K_m for prev_feat
        self.proj_in_mask = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        # Output projection
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, prev_feat):
        """
        Args:
            x (Tensor): shape [B, C, H, W], main features
            prev_feat (Tensor): shape [B, C, H, W], used to compute a mask
        Returns:
            out (Tensor): shape [B, C, H, W], attended features
        """
        b, c, h, w = x.shape

        # 1) Q, K, V from input x
        qkv = self.proj_in(x)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # each [B, C, H, W]
        v = prev_feat * v

        # 2) Q_m, K_m from prev_feat, used to compute the mask
        qk_m = self.proj_in_mask(prev_feat)  # [B, 2*C, H, W]
        q_m, k_m = qk_m.chunk(2, dim=1)      # each [B, C, H, W]

        # 3) Reshape for multi-head attention
        #    [B, (head*C), H, W] -> [B, head, C, H*W]
        q = rearrange(q,   'b (head c) h w -> b head c (h w)',   head=self.num_heads)
        k = rearrange(k,   'b (head c) h w -> b head c (h w)',   head=self.num_heads)
        v = rearrange(v,   'b (head c) h w -> b head c (h w)',   head=self.num_heads)
        q_m = rearrange(q_m, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_m = rearrange(k_m, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 4) Normalize Q and K (optional but common)
        q  = torch.nn.functional.normalize(q,  dim=-1)
        k  = torch.nn.functional.normalize(k,  dim=-1)
        q_m = torch.nn.functional.normalize(q_m, dim=-1)
        k_m = torch.nn.functional.normalize(k_m, dim=-1)

        # 5) Compute main attention scores
        attn = (q @ k.transpose(-2, -1)) * self.temperature1  # shape: [B, head, N, N], N = H*W

        # 6) Compute mask from prev_feat
        prev_feat_mask = (q_m @ k_m.transpose(-2, -1)) * self.temperature2  # [B, head, N, N]

        # 7) Convert prev_feat_mask to [0,1] range using sigmoid
        #    Optionally apply a threshold or more advanced logic if needed
        mask_0_1 = torch.sigmoid(prev_feat_mask)  # [B, head, N, N]

        # 8) Mask out low values. Example: anything below 0.5 => -1e9
        #    or you can multiply attn by mask_0_1 (but masked_fill is more typical for discrete cut-off)
        attn = attn.masked_fill(mask_0_1 < 0.1, -1e9)

        # 9) Softmax over last dimension
        attn = attn.softmax(dim=-1)

        # 10) Weighted sum of V
        out = attn @ v  # shape: [B, head, C, N]
        
        # 11) Reshape out to [B, (head*C), H, W]
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        # 12) Final output projection
        out = self.proj_out(out)
        out = out.view(b, c, h, w)

        return out


##########################################################################
# RetinexFormer FFN  
class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x)
        return out

##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        # self.vss_block = BasicBlock(dim=in_channels, d_state=16, ssm_ratio=1, mlp_ratio=4, mlp_type='ffnv02') # No FFN = SS2D only

        # self.attention1 = R_Masked_R_MHSA(dim=in_channels, num_heads=num_heads)
        # self.attention1 = R_MHSA_MSEF(dim=in_channels, num_heads=num_heads)
        # self.attention = R_MHSA(dim=in_channels, num_heads=num_heads)
        self.attention = MHSA(dim=in_channels, num_heads=num_heads)

        self.norm1 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)
        self.norm2 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)
        # self.norm3 = LayerNorm(dim=in_channels, LayerNorm_type=LayerNorm_type)

        self.ffn = FFN(dim=in_channels, mult=4)

        # self.alpha_param = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, prev_feat):
        # prev_feat = self.vss_block(prev_feat) # Attn + Post-op Norm + Residual

        # x = x + self.norm1(self.attention1(x, prev_feat)) # Attn + Norm + Residual

        x = x + self.norm1(self.attention(x))
        # x = x + self.norm1(self.attention(x, prev_feat))

        # x = self.norm2(x1 + x2)
        
        x = x + self.norm2(self.ffn(x))
        # x = self.vss_block(x)

        return x
## Model
# Lv2. - current - higher-dims
#Lv3. current + lower-dims
# V - CSE

class TransformerU(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, num_blocks=None, stage=None):
        super(TransformerU, self).__init__()
        
        num_heads = 2
        
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=1, padding='same')

        # First level
        self.transformer_block1_1 = TransformerBlock(n_feat, num_heads)
        self.downsample1 = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)
        # num_heads *=2
        # Second level
        self.transformer_block2_1 = TransformerBlock(n_feat * 2, num_heads)
        self.transformer_block2_2 = TransformerBlock(n_feat * 2, num_heads)
        self.downsample2 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)
        num_heads *=2
        # Bottleneck level
        self.bottleneck_1 = TransformerBlock(n_feat * 4, num_heads)
        self.bottleneck_2 = TransformerBlock(n_feat * 4, num_heads)
        num_heads //=2
        # Second level (upsampling)
        self.upsample2 = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust2 = nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1)
        self.transformer_block_up2_1 = TransformerBlock(n_feat * 2, num_heads)
        self.transformer_block_up2_2 = TransformerBlock(n_feat * 2, num_heads)
        num_heads //=2
        # First level (upsampling)
        self.upsample1 = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.channel_adjust1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)
        self.transformer_block_up1_1 = TransformerBlock(n_feat, num_heads)

        # Final output convolution
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.conv_in(x)
        # Downward path
        x1 = self.transformer_block1_1(x, x)
        x1_down = self.downsample1(x1)

        x2 = self.transformer_block2_1(x1_down, x1_down)
        x2 = self.transformer_block2_2(x2, x1_down)
        x2_down = self.downsample2(x2)

        # Bottleneck
        bn = self.bottleneck_1(x2_down, x2_down)
        bn = self.bottleneck_2(bn, x2_down)

        # Upward path
        
        x2_up = self.upsample2(bn)
        x2_up = torch.cat([x2_up, x2], dim=1)
        x2_up = self.channel_adjust2(x2_up)
        res = x2_up
        x2_up = self.transformer_block_up2_1(x2_up, res)
        x2_up = self.transformer_block_up2_2(x2_up, res)

        
        x1_up = self.upsample1(x2_up)
        x1_up = torch.cat([x1_up, x1], dim=1)
        x1_up = self.channel_adjust1(x1_up)
        res = x1_up
        x1_up = self.transformer_block_up1_1(x1_up, res)

        # Final output
        x_out = self.conv_out(x1_up)
        return x_out

class Transformer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=64, num_blocks=[1, 2, 2]):
        super(Transformer, self).__init__()
        print('CSE Arch Used.')
        self.num_blocks = num_blocks  # List specifying number of blocks at each level [D1, D2, D4]
        num_heads = 4
        self.n_feat = n_feat

        # Input projection
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=1, padding='same')

        # Downsampling layers with feature expansion
        self.downsample1 = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)
        self.downsample1_2 = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)  # For second level
        self.downsample2 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)
        self.downsample2_2 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)  # For second level
        self.downsample2_3 = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)  # For second level

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1_2 = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1_3 = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2_2 = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2_3 = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Transformer Blocks at scale D1 (input resolution)
        self.transformer_blocks_D1_1 = nn.ModuleList([
            TransformerBlock(n_feat, num_heads) for _ in range(num_blocks[0])
        ])

        # Transformer Blocks at scale D2
        self.transformer_blocks_D2_1 = nn.ModuleList([
            TransformerBlock(n_feat * 2, num_heads // 2) for _ in range(num_blocks[1])
        ])

        # Transformer Blocks at scale D4
        self.transformer_blocks_D4_1 = nn.ModuleList([
            TransformerBlock(n_feat * 4, num_heads // 2) for _ in range(num_blocks[2])
        ])

        # Second-Level Transformer Blocks
        self.transformer_blocks_D1_2 = nn.ModuleList([
            TransformerBlock(n_feat, num_heads) for _ in range(num_blocks[0])
        ])

        self.transformer_blocks_D2_2 = nn.ModuleList([
            TransformerBlock(n_feat * 2, num_heads // 2) for _ in range(num_blocks[1])
        ])

        self.transformer_blocks_D4_2 = nn.ModuleList([
            TransformerBlock(n_feat * 4, num_heads // 2) for _ in range(num_blocks[2])
        ])

        # Third-Level Transformer Blocks
        self.transformer_blocks_D1_3 = nn.ModuleList([
            TransformerBlock(n_feat, num_heads) for _ in range(num_blocks[0])
        ])

        self.transformer_blocks_D2_3 = nn.ModuleList([
            TransformerBlock(n_feat * 2, num_heads // 2) for _ in range(num_blocks[1])
        ])

        self.transformer_blocks_D4_3 = nn.ModuleList([
            TransformerBlock(n_feat * 4, num_heads // 2) for _ in range(num_blocks[2])
        ])

        # Final output convolution
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        
        # Input projection
        x_in = self.conv_in(x)  # [B, n_feat, H, W]

        # First Level Transformer Blocks at D1
        I_prime = x_in
        for block in self.transformer_blocks_D1_1:
            I_prime = block(I_prime, x_in)  # [B, n_feat, H, W]

        # Downsample to D2
        x_D2 = self.downsample1(I_prime)  # [B, n_feat * 2, H/2, W/2]

        # First Level Transformer Blocks at D2
        I2_prime = x_D2
        for block in self.transformer_blocks_D2_1:
            I2_prime = block(I2_prime, x_D2)  # [B, n_feat * 2, H/2, W/2]

        # Downsample to D4
        x_D4 = self.downsample2(I2_prime)  # [B, n_feat * 4, H/4, W/4]

        # First Level Transformer Blocks at D4
        I4_prime = x_D4
        for block in self.transformer_blocks_D4_1:
            I4_prime = block(I4_prime, x_D4)  # [B, n_feat * 4, H/4, W/4]

        # Second Level Transformer Blocks at D1
        I_double_prime = I_prime
        for block in self.transformer_blocks_D1_2:
            I_double_prime = block(I_double_prime, I_prime + x_in)  # [B, n_feat, H, W]

        # Prepare for I2''
        I_prime_D2 = self.downsample1_2(I_prime)  # [B, n_feat * 2, H/2, W/2]
        I2_input = I2_prime + I_prime_D2

        # Second Level Transformer Blocks at D2
        I2_double_prime = I2_input
        for block in self.transformer_blocks_D2_2:
            I2_double_prime = block(I2_double_prime, I2_input + x_D2)  # [B, n_feat * 2, H/2, W/2]

        # Prepare for I4''
        I2_prime_D2 = self.downsample2_2(I2_prime)  # [B, n_feat * 4, H/4, W/4]
        I_prime_D4 = self.downsample2_3(I_prime_D2)  # [B, n_feat * 4, H/4, W/4]
        I4_input = I4_prime + I2_prime_D2 + I_prime_D4

        # Second Level Transformer Blocks at D4
        I4_double_prime = I4_input
        for block in self.transformer_blocks_D4_2:
            I4_double_prime = block(I4_double_prime, I4_input + x_D4)  # [B, n_feat * 4, H/4, W/4]

        # Third Level Transformer Blocks at D1
        # Upsample I2_double_prime and I4_double_prime to D1
        I2_double_prime_U = self.upsample1(I2_double_prime)  # [B, n_feat, H, W]
        I4_double_prime_U = self.upsample1_2(self.upsample2(I4_double_prime))  # [B, n_feat, H, W]
        I_triple_prime_input = I_double_prime + I2_double_prime_U + I4_double_prime_U

        I_triple_prime = I_triple_prime_input
        for block in self.transformer_blocks_D1_3:
            I_triple_prime = block(I_triple_prime, I_triple_prime_input + x_in + I_prime)  # [B, n_feat, H, W]

        # Third Level Transformer Blocks at D2
        # Upsample I4_double_prime to D2
        I4_double_prime_U2 = self.upsample2_2(I4_double_prime)  # [B, n_feat * 2, H/2, W/2]
        I2_triple_prime_input = I2_double_prime + I4_double_prime_U2

        I2_triple_prime = I2_triple_prime_input
        for block in self.transformer_blocks_D2_3:
            I2_triple_prime = block(I2_triple_prime, I2_triple_prime_input + I2_input + x_D2)  # [B, n_feat * 2, H/2, W/2]

        # Third Level Transformer Blocks at D4
        I4_triple_prime = I4_double_prime
        for block in self.transformer_blocks_D4_3:
            I4_triple_prime = block(I4_triple_prime, I4_double_prime + I4_input + x_D4)  # [B, n_feat * 4, H/4, W/4]

        # Final Output
        # Upsample I2_triple_prime and I4_triple_prime to D1
        I2_triple_prime_U = self.upsample1_3(I2_triple_prime)  # [B, n_feat, H, W]
        I4_triple_prime_U = self.upsample1_2(self.upsample2_3(I4_triple_prime))  # [B, n_feat, H, W]
        output = I_triple_prime + I2_triple_prime_U + I4_triple_prime_U  # [B, n_feat, H, W]
        scalar = (1/3 * ((I_triple_prime - I2_triple_prime_U)**2 
                    + (I_triple_prime - I4_triple_prime_U)**2 
                    + (I2_triple_prime_U - I4_triple_prime_U)**2)).mean()
        print(scalar.item())
        # Final output convolution
        x_out = self.conv_out(output)  # [B, out_channels, H, W]

        return x_out