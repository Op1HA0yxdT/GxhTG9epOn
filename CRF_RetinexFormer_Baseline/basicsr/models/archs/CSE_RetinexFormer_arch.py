import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]



class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map



class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
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
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

        # ---- visualization controls (do NOT affect weights) ----
        self.enable_vis: bool = False
        self._vis_dir: str | None = None
        self._vis_idx: int = 0

    # -------------------------------------------------------------
    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)

        # optional visualization of final featuremap
        if self.enable_vis:
            with torch.no_grad():
                self._save_featuremap_viz(out, prefix="igab_out")

        return out

    # -------------------------------------------------------------
    def _ensure_vis_dir(self):
        if self._vis_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._vis_dir = os.path.join(os.getcwd(), f"igab_featuremaps_{ts}")
            os.makedirs(self._vis_dir, exist_ok=True)

    def _reduce_to_2d(self, feat: torch.Tensor) -> torch.Tensor:
        """Reduce (B,C,H,W) → (B,H,W) using L2 norm across channels."""
        return torch.norm(feat, p=2, dim=1)

    def _normalize01(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Per-sample min-max normalization."""
        B = x.shape[0]
        x_flat = x.view(B, -1)
        mn = x_flat.min(dim=1, keepdim=True).values
        mx = x_flat.max(dim=1, keepdim=True).values
        return ((x_flat - mn) / (mx - mn + eps)).view_as(x)

    def _save_featuremap_viz(self, feat: torch.Tensor, prefix: str = "feat"):
        """
        Saves featuremap visualization using the green-blue colormap ('winter').
        One PNG per batch item.
        """
        self._ensure_vis_dir()
        self._vis_idx += 1
        feat_cpu = feat.detach().float().cpu()

        fmap = self._reduce_to_2d(feat_cpu)
        fmap = self._normalize01(fmap)
        fmap = 1 - fmap

        for b in range(fmap.shape[0]):
            arr = fmap[b].numpy()
            plt.figure(figsize=(4, 4), dpi=150)
            plt.axis('off')
            plt.imshow(arr, cmap='hot', vmin=0.0, vmax=1.0)
            fname = f"{prefix}_{self._vis_idx:06d}_b{b}.png"
            fpath = os.path.join(self._vis_dir, fname)
            plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
            plt.close()



class _Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(_Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """
        # print(x.shape, illu_fea.shape)
        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

    
class Denoiser(nn.Module):
    def __init__(
        self, 
        in_dim=3, 
        out_dim=3, 
        n_feat=64, 
        num_blocks=[1, 2, 2]  # e.g. scale D1 uses 1 block, D2 uses 2, D4 uses 2
    ):
        super(Denoiser, self).__init__()

        self.num_blocks = num_blocks
        self.n_feat = n_feat

        # --------------------------------------------------
        # 1) Input Projections
        # --------------------------------------------------
        # For the main image feature:
        self.conv_in = nn.Conv2d(in_dim, n_feat, kernel_size=3, padding=1, bias=False)
        self.conv_in_illu = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=False)

        # --------------------------------------------------
        # 2) Downsampling Layers 
        # --------------------------------------------------
        # First time we go from D1 -> D2
        self.downsample1_x    = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)
        self.downsample1_illu = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)

        # Second time we go from D2 -> D4
        self.downsample2_x    = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)
        self.downsample2_illu = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)

        self.downsample1_2_x    = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)
        self.downsample1_2_illu = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1)

        self.downsample2_2_x    = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)
        self.downsample2_2_illu = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)

        self.downsample2_3_x    = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)
        self.downsample2_3_illu = nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=3, stride=2, padding=1)

        # --------------------------------------------------
        # 3) Upsampling Layers
        # --------------------------------------------------
        self.upsample1_x    = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1_illu = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample1_2_x    = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1_2_x_    = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample1_2_illu = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample1_3_x    = nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample2_x    = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2_x_    = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2_illu = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2_illu_ = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.upsample2_3_x    = nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1)

        # --------------------------------------------------
        # 4) Transformer Blocks

        dim_level = n_feat
        self.transformer_blocks_D1_1 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[0])
        ])
        # Round 1 at D2
        dim_level *= 2
        self.transformer_blocks_D2_1 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[1])
        ])
        # Round 1 at D4
        dim_level *= 2
        self.transformer_blocks_D4_1 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[2])
        ])

        # Round 2 at D1
        dim_level //= 4
        self.transformer_blocks_D1_2 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[0])
        ])
        # Round 2 at D2
        dim_level *= 2
        self.transformer_blocks_D2_2 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[1])
        ])
        # Round 2 at D4
        dim_level *= 2
        self.transformer_blocks_D4_2 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[2])
        ])

        # Round 3 at D1
        dim_level //= 4
        self.transformer_blocks_D1_3 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[0])
        ])
        # Round 3 at D2
        dim_level *= 2
        self.transformer_blocks_D2_3 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[1])
        ])
        # Round 3 at D4
        dim_level *= 2
        self.transformer_blocks_D4_3 = nn.ModuleList([
            IGAB(dim=dim_level, num_blocks=1, heads=dim_level // n_feat, dim_head=n_feat) for _ in range(num_blocks[2])
        ])

        # --------------------------------------------------
        # 5) Output Projection
        # --------------------------------------------------
        self.conv_out = nn.Conv2d(n_feat, out_dim, kernel_size=3, padding=1, bias=False)

        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:        [B, in_dim,  H, W]
        illu_fea: [B, in_dim,  H, W]
        return:   [B, out_dim, H, W]
        """
        # --------------------------------------------------
        # 1) Project Inputs
        # --------------------------------------------------
        x_in      = self.conv_in(x)           # [B, n_feat, H,   W]
        illu_in   = self.conv_in_illu(illu_fea)  # [B, n_feat, H,   W]

        # --------------------------------------------------
        # 2) Round 1: D1 -> D2 -> D4
        # --------------------------------------------------
        # --- D1 Blocks ---
        I_prime = x_in
        for block in self.transformer_blocks_D1_1:
            I_prime = block(I_prime, illu_in)  # IGAB(fea, illu_fea)

        # Downsample to D2
        x_D2     = self.downsample1_x(x_in)      # [B, n_feat*2, H/2, W/2]
        illu_D2  = self.downsample1_illu(illu_in)   # [B, n_feat*2, H/2, W/2]

        # --- D2 Blocks ---
        I2_prime = x_D2
        for block in self.transformer_blocks_D2_1:
            I2_prime = block(I2_prime, illu_D2)

        # Downsample to D4
        x_D4     = self.downsample2_x(x_D2)     # [B, n_feat*4, H/4, W/4]
        illu_D4  = self.downsample2_illu(illu_D2)   # [B, n_feat*4, H/4, W/4]

        # --- D4 Blocks ---
        I4_prime = x_D4
        for block in self.transformer_blocks_D4_1:
            I4_prime = block(I4_prime, illu_D4)

        # --------------------------------------------------
        # 3) Round 2: re-aggregate at D1, D2, D4
        # --------------------------------------------------
        # ---------- D1 ------------
        I_double_prime = I_prime
        for block in self.transformer_blocks_D1_2:
            I_double_prime = block(I_double_prime, illu_in)

        # Prepare for I2'' 
        I_prime_D2    = self.downsample1_2_x(I_prime)   # [B, n_feat*2, H/2, W/2]
        illu_prime_D2 = self.downsample1_2_illu(illu_in)
        I2_input = I2_prime + I_prime_D2  # entangle them

        # ---------- D2 ------------
        I2_double_prime = I2_input
        for block in self.transformer_blocks_D2_2:
            I2_double_prime = block(I2_double_prime, illu_prime_D2 + illu_D2)

        # Prepare for I4''
        I2_prime_D4     = self.downsample2_2_x(I2_prime)    # [B, n_feat*4, H/4, W/4]
        illu2_prime_D4  = self.downsample2_2_illu(illu_D2)  
        I_prime_D4      = self.downsample2_3_x(I_prime_D2)  # [B, n_feat*4, H/4, W/4]
        illu_prime_D4_2 = self.downsample2_3_illu(illu_prime_D2)

        I4_input = I4_prime + I2_prime_D4 + I_prime_D4
        illu4_input = illu_D4 + illu2_prime_D4 + illu_prime_D4_2

        # ---------- D4 ------------
        I4_double_prime = I4_input
        for block in self.transformer_blocks_D4_2:
            I4_double_prime = block(I4_double_prime, illu4_input)

        # --------------------------------------------------
        # 4) Round 3: final re-aggregation
        # --------------------------------------------------
        # ---------- D1 ------------
        # Upsample from D2
        I2_double_prime_U = self.upsample1_x(I2_double_prime) 
        illu2_double_prime_U = self.upsample1_illu(illu_prime_D2)

        # Upsample from D4
        I4_double_prime_U = self.upsample1_2_x_(self.upsample2_x_(I4_double_prime))
        illu4_double_prime_U = self.upsample1_2_illu(self.upsample2_illu_(illu4_input))

        # Combine
        I_triple_prime_input = I_double_prime + I2_double_prime_U + I4_double_prime_U
        illu_triple_prime_input = illu_in + illu2_double_prime_U + illu4_double_prime_U

        I_triple_prime = I_triple_prime_input
        for block in self.transformer_blocks_D1_3:
            I_triple_prime = block(I_triple_prime, illu_triple_prime_input)

        # ---------- D2 ------------
        # Upsample I4_double_prime to D2
        I4_double_prime_U2 = self.upsample2_x(I4_double_prime) 
        illu4_double_prime_U2 = self.upsample2_illu(illu4_input)

        I2_triple_prime_input = I2_double_prime + I4_double_prime_U2
        illu2_triple_prime_input = illu_prime_D2 + illu4_double_prime_U2

        I2_triple_prime = I2_triple_prime_input
        for block in self.transformer_blocks_D2_3:
            # print('I2_triple_prime', I2_triple_prime.shape, 'illu2_double_prime_U', illu4_double_prime_U2.shape)
            I2_triple_prime = block(I2_triple_prime, illu2_triple_prime_input)

        # ---------- D4 ------------
        I4_triple_prime = I4_double_prime
        for block in self.transformer_blocks_D4_3:
            # print('I4_triple_prime', I4_triple_prime.shape, 'illu4_input', illu4_input.shape)
            I4_triple_prime = block(I4_triple_prime, illu4_input if 'illu4_input' in locals() else illu4_input)

        # --------------------------------------------------
        # 5) Final Output at D1 
        # --------------------------------------------------
        # Upsample from D2
        I2_triple_prime_U = self.upsample1_3_x(I2_triple_prime)
        # Upsample from D4
        I4_triple_prime_U = self.upsample1_2_x(self.upsample2_3_x(I4_triple_prime))

        # Combine
        output = I_triple_prime + I2_triple_prime_U + I4_triple_prime_U
        x_out  = self.conv_out(output)

        return x_out + x
    


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels,out_dim=out_channels,n_feat=n_feat,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img
        # self.denoiser = _Denoiser(in_dim=in_channels,out_dim=out_channels,dim=n_feat,level=level,num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, img):
        # img:        b,c=3,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        output_img = self.denoiser(input_img,illu_fea)

        return output_img


class CSE_RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1,1,1]):
        super(CSE_RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks)
                        for _ in range(stage)]
        
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)
        return out
    
from fvcore.nn import FlopCountAnalysis
if __name__ == "__main__":

    ##Calculate Parameters and FLOPs
    device = 'cuda:0'
    model = CSE_RetinexFormer(in_channels=3, out_channels=3, n_feat=24, stage=1, num_blocks=[1,2,2]).to(device)
    model.eval()

    input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        # flops, params = profile(model, inputs=(input1, (input2,input3)))
        # print("FLOPs=", str(flops / (1024*1024*1024)) + '{}'.format("G"))
        # print("params=", str(params / (1024*1024)) + '{}'.format("M"))

        flops = FlopCountAnalysis(model, inputs=input)
        n_param = sum([p.nelement() for p in model.parameters()])
        print("FLOPs=", str(flops.total() / (1024 * 1024 * 1024)) + '{}'.format("G"))
        print("params=", str(n_param / (1024 * 1024)) + '{}'.format("M"))