"""ProGIC inference architecture with checkpoint-compatible parameter names."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def conv(in_ch: int, out_ch: int, kernel_size: int = 5, stride: int = 1, bias = True, padding = None, groups = 1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding if padding is not None else kernel_size // 2, bias = bias, groups=groups)

def relu():
    return nn.ReLU(inplace=True)

class ReLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = relu()

    def forward(self, x):
        x1, x2 = self.relu(x).chunk(2, 1)
        return x1 + x2

class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            conv(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = conv(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            conv(out_ch, out_ch, 1),
            relu(),
            conv(out_ch, out_ch, 3, padding=1, groups=out_ch),
            conv(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            conv(out_ch, out_ch * 4, 1),
            ReLUChunkAdd(),
            conv(out_ch * 2, out_ch, 1),
        )
        self.proxy = None

    def forward(self, x):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out

class AttentionBlock(nn.Module):
    """Self attention block.
    Modified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.
    """

    def __init__(self, N: int):
        super().__init__()

        self.conv_a = nn.Sequential(
            DepthConvBlock(N, N),
            DepthConvBlock(N, N),
            DepthConvBlock(N, N),
        )

        self.conv_b = nn.Sequential(
            DepthConvBlock(N, N),
            DepthConvBlock(N, N),
            DepthConvBlock(N, N),
            conv(N, N, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class DepthConvBlockScale(nn.Module):
    def __init__(self, in_ch, out_ch, scale_num, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            relu(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            ReLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )
        self.scale1 = nn.Parameter(torch.zeros(scale_num, out_ch, 1, 1))
        self.scale2 = nn.Parameter(torch.zeros(scale_num, out_ch, 1, 1))

        self.bias1 = nn.Parameter(torch.zeros(scale_num, out_ch, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(scale_num, out_ch, 1, 1))
        self.proxy = None

    def forward(self, x, scale_ind = None):
        if self.adaptor is not None:
            x = self.adaptor(x)
        if scale_ind is not None:
            scale = self.scale1[scale_ind:scale_ind+1, :, :, :]
            bias = self.bias1[scale_ind:scale_ind+1, :, :, :]
            out = self.dc(x) * (scale+1) + x + bias
        else:
            out = self.dc(x) + x

        if scale_ind is not None:
            scale = self.scale2[scale_ind:scale_ind+1, :, :, :]
            bias = self.bias2[scale_ind:scale_ind+1, :, :, :]
            out = self.ffn(out) * (scale+1) + out + bias
        else:
            out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out

class ResidualBlockUpsampleScale(nn.Module):
    def __init__(self, in_ch, out_ch, scale_num):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlockScale(out_ch, out_ch, scale_num = scale_num, shortcut=True)

    def forward(self, x, scale_num):
        out = self.up(x)
        out = self.conv(out, scale_num)
        return out

class AttentionBlockScale(nn.Module):
    """Self attention block.
    Modified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.
    """

    def __init__(self, N: int, scale_num: int):
        super().__init__()

        self.conv_a = nn.Sequential(
            DepthConvBlockScale(N, N, scale_num),
            DepthConvBlockScale(N, N, scale_num),
            DepthConvBlockScale(N, N, scale_num),
        )

        self.conv_b1 = nn.Sequential(
            DepthConvBlockScale(N, N, scale_num),
            DepthConvBlockScale(N, N, scale_num),
            DepthConvBlockScale(N, N, scale_num),
        )

        self.conv_b2 = nn.Sequential(
            nn.Conv2d(N, N, 1),
        )

    def forward(self, x: Tensor, scale_ind: int) -> Tensor:
        identity = x
        a = x
        b = x
        for conv in self.conv_a:
            a = conv(a, scale_ind)
        for conv in self.conv_b1:
            b = conv(b, scale_ind)
        b = self.conv_b2(b)
        out = a * torch.sigmoid(b)
        out += identity
        return out

class Quantizer(nn.Module):
    """One RVQ stage. Spatial dimensions travel with the index tensor."""

    def __init__(self, n_e=1024, e_dim=8, in_dim=256):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        # Keep legacy weight_norm parameter names for the released checkpoints.
        self.in_proj = nn.utils.weight_norm(nn.Conv2d(in_dim, e_dim, 1))
        self.embedding = nn.Embedding(n_e, e_dim)
        self.out_proj = nn.utils.weight_norm(nn.Conv2d(e_dim, in_dim, 1))

    def encoding(self, z):
        z = self.in_proj(z).permute(0, 2, 3, 1).contiguous()
        flat = z.reshape(-1, self.e_dim)
        distances = (
            flat.square().sum(1, keepdim=True)
            + self.embedding.weight.square().sum(1)
            - 2 * torch.einsum('bd,dn->bn', flat, self.embedding.weight.T)
        )
        return distances.argmin(1).reshape(z.shape[:3])

    def decoding(self, indices):
        z = self.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.out_proj(z)


class ResidualQuantize(nn.Module):
    def __init__(self, n_codebooks=5, n_e=1024, e_dim=8, in_dim=256):
        super().__init__()
        self.quantizers = nn.ModuleList([
            Quantizer(n_e, e_dim, in_dim) for _ in range(n_codebooks)
        ])

    def encoding(self, z, num_stages):
        indices = []
        residual = z
        for quantizer in self.quantizers[:num_stages]:
            stage = quantizer.encoding(residual)
            residual = residual - quantizer.decoding(stage)
            indices.append(stage)
        return indices

    def decoding(self, indices):
        z = self.quantizers[0].decoding(indices[0])
        for quantizer, stage in zip(self.quantizers[1:], indices[1:]):
            z = z + quantizer.decoding(stage)
        return z


class IntraEncoder(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.enc_1 = nn.Sequential(nn.Conv2d(192, channels, 1))
        self.enc_2 = nn.Sequential(
            *[DepthConvBlock(channels, channels) for _ in range(depth)],
            conv(channels, 256, 3, stride=2),
            AttentionBlock(256),
        )

    def forward(self, x):
        return self.enc_2(self.enc_1(F.pixel_unshuffle(x, 8)))


class IntraDecoder(nn.Module):
    def __init__(self, channels, depth):
        super().__init__()
        self.bias1 = nn.Parameter(torch.zeros(5, 256, 1, 1))
        self.dec_1 = nn.Sequential(
            AttentionBlockScale(256, 5),
            ResidualBlockUpsampleScale(256, channels, 5),
            *[DepthConvBlockScale(channels, channels, 5) for _ in range(depth)],
        )
        self.dec_2 = conv(channels, 192, 1, 1)

    def forward(self, x, index):
        x = x + self.bias1[index:index + 1]
        for layer in self.dec_1:
            x = layer(x, index)
        return F.pixel_shuffle(self.dec_2(x), 8)


class ProGIC(nn.Module):
    """Inference model. index=0..4 uses 1..5 residual codebooks.

    Inputs are RGB tensors in [-1, 1], with height and width divisible by 16.
    compress returns a list of integer tensors with shape (B, H/16, W/16).
    A separate decoder with the same weights can decode any nonempty prefix.
    """

    def __init__(self, variant="base"):
        super().__init__()
        settings = {"base": (368, 8, 14), "small": (256, 4, 8)}
        channels, encoder_depth, decoder_depth = settings[variant]
        self.g_a = IntraEncoder(channels, encoder_depth)
        self.g_s = IntraDecoder(channels, decoder_depth)
        self.quantize = ResidualQuantize()
        self.num_stages = 5
        self.bits_per_index = 10

    @torch.inference_mode()
    def compress(self, x, index=4):
        if index not in range(self.num_stages):
            raise ValueError("index must be in 0..4")
        if x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Pad image height and width to multiples of 16 before encoding")
        return self.quantize.encoding(self.g_a(x), index + 1)

    @torch.inference_mode()
    def decompress(self, indices, index=None):
        if index is None:
            index = len(indices) - 1
        if index not in range(self.num_stages) or len(indices) < index + 1:
            raise ValueError("Decoding requires index+1 codebook tensors, with index in 0..4")
        y = self.quantize.decoding(indices[:index + 1])
        return self.g_s(y, index)

    @torch.inference_mode()
    def forward(self, x, index=4):
        return self.decompress(self.compress(x, index), index)
