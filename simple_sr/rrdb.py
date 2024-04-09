import functools
from torch import nn
from .commons import RRDB


class RRDBNet(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel, num_blocks, gc=32):
        super(RRDBNet, self).__init__()

        RRDB_block_f = functools.partial(RRDB, nf=mid_channel, gc=gc)

        self.conv_first = nn.Conv2d(in_channel, mid_channel, kernel_size=3,
                                    padding=1, padding_mode='reflect')
        # self.RRDB_trunk = make_layer(RRDB_block_f, num_blocks)
        
        layers = []
        for _ in range(num_blocks):
            layers.append(RRDB_block_f())

        self.RRDB_trunk = nn.ModuleList(layers)
        
        self.trunk_conv = nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
                                    padding=1, padding_mode='reflect')

        # # upsampling
        # self.upconv1 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
        #                          padding=1, padding_mode='reflect')
        # self.upconv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
        #                          padding=1, padding_mode='reflect')
        # self.HRconv = nn.Conv2d(mid_channel, mid_channel, kernel_size=3,
        #                         padding=1, padding_mode='reflect')
        self.conv_last = nn.Conv2d(mid_channel, out_channel, kernel_size=3,
                                   padding=1, padding_mode='reflect')

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        # x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)
        for layer in self.RRDB_trunk:
            fea = layer(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        # fea = self.lrelu(self.upconv1(fea))
        # fea = self.lrelu(self.upconv2(fea))

        # fea_hr = self.HRconv(fea)
        # out = self.conv_last(self.lrelu(fea_hr))
        out = self.conv_last(fea)
        # out = out.clamp(0, 1)
        # out = out * 2 - 1

        if get_fea:
            return out, feas
        else:
            return out