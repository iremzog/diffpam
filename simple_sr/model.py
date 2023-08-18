import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        if groups == 0:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                Mish()
            )
        else:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                nn.GroupNorm(groups, dim_out),
                Mish()
            )

    def forward(self, x):
        return self.block(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(
                Mish(),
                nn.Linear(time_emb_dim, dim_out)
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if time_emb is not None:
            h += self.mlp(time_emb)[:, :, None, None]
        if cond is not None:
            h += cond
        h = self.block2(h)
        return h + self.res_conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 2),
        )

    def forward(self, x):
        return self.conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

'''
    U-Net model to be used in PSNR based methods
'''


class SimpleSRUNet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        self.first_conv = Block(out_dim, dim, groups=0)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, groups=groups),
                ResnetBlock(dim_out, dim_out, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, groups=groups)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, groups=groups),
                ResnetBlock(dim_in, dim_in, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x):

        h = []
        x = self.first_conv(x)

        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x)
            x = resnet2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = upsample(x)

        return self.final_conv(x)