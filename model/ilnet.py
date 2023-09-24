from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, ceil, floor


class ConvBNReLU(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.relu(self.bn(self.conv(x)))


class UpConvBNReLU(ConvBNReLU):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x1, x2], dim=1))))


class RSU(nn.Module):
    """ Residual U-block """

    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)  # stem
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))

        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class IPOF(nn.Module):

    def __init__(self, in_channel, kernel_size=1, stride=1):
        super(IPOF, self).__init__()

        self.in_channel = in_channel
        self.inter_channel = in_channel // 2
        self.out_channel = in_channel * 2
        self.kernel_size = kernel_size
        self.stride = stride
        ratio = 4

        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, 1, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_channel // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channel // ratio, self.in_channel, kernel_size=1),
            nn.LayerNorm([self.in_channel, 1, 1]),

        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU(True))
        self.DODA = DODA(self.inter_channel)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Sequential(
            # nn.Conv2d(self.in_channel, self.in_channel, 3, 1, 1),
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(True)
        )

    def channel(self, E, D):
        D_ = self.conv_2(D)
        batch, channel, height, width = D_.size()
        D_ = D_.view(batch, channel, height * width)  # [N, IC, H*W]

        E = self.conv_1(E)  # [N, 1, H, W]
        E = E.view(batch, 1, height * width)  # [N, 1, H*W]
        E = self.softmax(E)  # [N, 1, H*W]

        context = torch.matmul(D_, E.transpose(1, 2)).unsqueeze(-1)  # [N, IC, 1, 1]
        context = self.conv_up(self.DODA(context))  # [N, OC, 1, 1]

        out = D * self.sigmoid(context)  # [N, OC, 1, 1]
        return out

    def spatial(self, E, D):
        E_ = self.conv_3(E)  # [N, IC, H, W]
        batch, channel, height, width = E_.size()
        E_ = self.DODA(self.avg_pool(E_))  # [N, IC, 1, 1]
        batch, channel, avg_e_h, avg_e_w = E_.size()
        E_ = E_.view(batch, channel, avg_e_h * avg_e_w).permute(0, 2, 1)  # [N, 1, IC]
        E_ = self.softmax(E_)

        D = self.conv_4(D).view(batch, self.inter_channel, height * width)  # [N, IC, H*W]

        context = torch.matmul(E_, D).view(batch, 1, height, width)  # [N, 1, H, W]
        context = F.layer_norm(context, normalized_shape=(1, context.shape[-2], context.shape[-1]))

        out = E * self.sigmoid(context)  # [N, 1, H, W]
        return out

    def forward(self, E, D):
        channel = self.channel(E, D)  # [N, C, H, W]
        spatial = self.spatial(E, D)  # [N, C, H, W]
        out = self.out(spatial + channel)  # [N, C, H, W]
        return out


class DODA(nn.Module):
    def __init__(self, channel, n=2, b=2, inter='ceil'):
        super(DODA, self).__init__()

        kernel_size = int(abs((log(channel, 2) + 1) / 2))
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.layer_size = 1 + (log(channel, 2) - b) / (2 * n)
        assert inter == 'ceil' or 'floor'
        if inter == 'ceil':
            self.layer_size = ceil(self.layer_size)
        else:
            self.layer_size = floor(self.layer_size)

        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=int(self.kernel_size / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = self.conv(x.squeeze(-1).transpose(-1, -2))
        for i in range(self.layer_size):
            h = self.conv(h)
        h = h.transpose(-1, -2).unsqueeze(-1)

        h = self.sigmoid(h)
        return x + h


class ILNet(nn.Module):

    def __init__(self, cfg: dict, out_ch: int = 1, rb=True):
        super().__init__()
        assert "encode" and "decode" in cfg
        self.encode_num = len(cfg["encode"])
        self.rb = rb
        encode_list = []
        loss_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                if self.rb:
                    loss_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        ipof_list = []
        for i, c in enumerate(cfg["decode"]):
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            ipof_list.append(IPOF(int(c[1] / 2)))

            if c[5] is True:
                if self.rb:
                    loss_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
                    side_list.append(nn.Conv2d(c[3], 2 ** i, kernel_size=3, padding=1))
                    self.side543 = nn.Conv2d(c[3], 1, kernel_size=3, padding=1)
                    self.side543_cat = nn.Conv2d(3, 1, kernel_size=1)
                else:
                    side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))

        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        if self.rb:
            self.loss_modules = nn.ModuleList(loss_list)
            self.out_conv = nn.Conv2d(32, out_ch, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(self.encode_num * out_ch, out_ch, kernel_size=1)

        self.ipof_modules = nn.ModuleList(ipof_list)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in zip(self.decode_modules, self.ipof_modules):
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m[1](x, x2)
            x = m[0](x)
            decode_outputs.insert(0, x)

        side_outputs = []
        loss_outputs = []

        # RB
        if self.rb:
            y3 = F.interpolate(self.side543(decode_outputs[2]), size=[512, 512], mode='bilinear', align_corners=False)
            y4 = F.interpolate(self.side543(decode_outputs[1]), size=[512, 512], mode='bilinear', align_corners=False)
            y5 = self.side543(decode_outputs[0])
            y = self.side543_cat(torch.concat([y3, y4, y5], dim=1))
            y = torch.sigmoid(y)
            for i, m in enumerate(zip(self.side_modules, self.loss_modules)):
                x = decode_outputs.pop()
                z = x.clone()
                z = F.interpolate(m[1](z), size=[h, w], mode='bilinear', align_corners=False)
                if i <= 2:
                    x = F.interpolate(m[0](x), size=[h, w], mode='bilinear', align_corners=False)
                    x = x * y
                else:
                    x = F.interpolate(m[0](x), size=[h, w], mode='bilinear', align_corners=False)
                side_outputs.insert(0, x)
                loss_outputs.insert(0, z)
        else:
            for m in self.side_modules:
                x = decode_outputs.pop()
                x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
                side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            if self.rb:
                return [x] + loss_outputs
            else:
                return [x] + side_outputs
        else:
            return self.bn(x)


def ILNet_L(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 32, 64, False, False],  # En3
                   [4, 64, 32, 128, False, False],  # En4
                   [4, 128, 32, 128, True, False],  # En5
                   [4, 128, 64, 128, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 256, 64, 128, True, True],  # De5
                   [4, 256, 32, 64, False, True],  # De4
                   [5, 128, 32, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return ILNet(cfg, out_ch, rb=False)


def ILNet_M(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return ILNet(cfg, out_ch)


def ILNet_S(out_ch: int = 1):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 4, 8, False, False],  # En1
                   [6, 8, 4, 8, False, False],  # En2
                   [5, 8, 4, 8, False, False],  # En3
                   [4, 8, 4, 8, False, False],  # En4
                   [4, 8, 4, 8, True, False],  # En5
                   [4, 8, 4, 8, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 16, 4, 8, True, True],  # De5
                   [4, 16, 4, 8, False, True],  # De4
                   [5, 16, 4, 8, False, True],  # De3
                   [6, 16, 4, 8, False, True],  # De2
                   [7, 16, 4, 8, False, True]]  # De1
    }

    return ILNet(cfg, out_ch)
