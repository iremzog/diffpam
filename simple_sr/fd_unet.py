import torch
from torch import nn
from .commons import Residual, Rezero, LinearAttention
from .commons import Upsample, Block, Downsample, FullyDenseBlock


class FDUnet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32):
        super().__init__()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0

        self.first_conv = Block(1, dim, groups=0)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                FullyDenseBlock(dim_in, dim_out),
                FullyDenseBlock(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = FullyDenseBlock(mid_dim, mid_dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = FullyDenseBlock(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                FullyDenseBlock(dim_out * 2, dim_in),
                FullyDenseBlock(dim_in, dim_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x):

        h = []
        x = self.first_conv(x)

        for i, (fd, fd2, downsample) in enumerate(self.downs):
            x = fd(x)
            x = fd2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for fd, fd2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = fd(x)
            x = fd2(x)
            x = upsample(x)

        return self.final_conv(x)