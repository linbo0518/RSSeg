import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class Interpolate(nn.Module):

    def __init__(self, scale_factor=2):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True)


class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.upconv = nn.Sequential(
            Interpolate(2),
            conv3x3(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.upconv(x)


class Decoder(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.upconv = UpConv(in_ch, out_ch)
        self.resblock = ResidualBlock(in_ch, out_ch, 1)

    def forward(self, d, e):
        d = self.upconv(d)
        d = torch.cat((d, e), dim=1)
        d = self.resblock(d)
        return d


class FPNBlock(nn.Module):

    def __init__(self, d0_in, d1_in, d2_in):
        super(FPNBlock, self).__init__()
        self.d0_output = nn.Sequential(conv3x3(d0_in, 1))
        self.d1_output = nn.Sequential(Interpolate(2), conv3x3(d1_in, 1))
        self.d2_output = nn.Sequential(Interpolate(4), conv3x3(d2_in, 1))
        self.final_output = conv3x3(3, 1)

    def forward(self, d0, d1, d2):
        d0 = self.d0_output(d0)
        d1 = self.d1_output(d1)
        d2 = self.d2_output(d2)
        x = torch.cat((d0, d1, d2), 1)
        final = self.final_output(x)
        return final


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(ResidualBlock, self).__init__()
        self.do_downsample = not (in_ch == out_ch and stride == 1)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if self.do_downsample:
            # self.conv3 = conv3x3(in_ch, out_ch, stride)
            self.conv3 = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.do_downsample:
            residual = self.conv3(residual)
            residual = self.bn3(residual)

        x += residual
        x = self.relu(x)
        return x


class ResNet34(nn.Module):

    def __init__(self):
        super(ResNet34, self).__init__()
        self.channel = 64
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage1 = self._add_stage(ResidualBlock, 64, 3)
        self.stage2 = self._add_stage(ResidualBlock, 128, 4)
        self.stage3 = self._add_stage(ResidualBlock, 256, 6)
        self.stage4 = self._add_stage(ResidualBlock, 512, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(512, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

    def _add_stage(self, block, channel, repeat):
        block_list = []
        stride = 1
        for _ in range(repeat):
            if self.channel != channel:
                stride = 2
            else:
                stride = 1
            block_list.append(block(self.channel, channel, stride))
            self.channel = channel
        return nn.Sequential(*block_list)


class ResUNet(nn.Module):

    def __init__(self, pretrained_params=None):
        super(ResUNet, self).__init__()
        self.encoder = ResNet34()
        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.upconv1 = UpConv(64, 32)
        self.upconv2 = UpConv(32, 16)
        self.out = conv3x3(16, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained_params is not None:
            self.encoder.load_state_dict(
                torch.load(pretrained_params), strict=False)
            print('pretrained model loaded')

    def forward(self, x):
        e0 = self.encoder.stage0(x)
        e1 = self.encoder.stage1(e0)
        e2 = self.encoder.stage2(e1)
        e3 = self.encoder.stage3(e2)
        c = self.encoder.stage4(e3)
        d3 = self.decoder3(c, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        d1_2 = self.upconv1(d1)
        d1_3 = self.upconv2(d1_2)
        out = self.out(d1_3)
        return out


class ResUNetFPN(nn.Module):

    def __init__(self, pretrained_params=None):
        super(ResUNetFPN, self).__init__()
        self.encoder = ResNet34()
        self.decoder3 = Decoder(512, 256)
        self.decoder2 = Decoder(256, 128)
        self.decoder1 = Decoder(128, 64)
        self.upconv1 = UpConv(64, 32)
        self.upconv2 = UpConv(32, 16)
        self.out = FPNBlock(16, 32, 64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained_params is not None:
            self.encoder.load_state_dict(
                torch.load(pretrained_params), strict=False)
            print('pretrained model loaded')

    def forward(self, x):
        e0 = self.encoder.stage0(x)
        e1 = self.encoder.stage1(e0)
        e2 = self.encoder.stage2(e1)
        e3 = self.encoder.stage3(e2)
        c = self.encoder.stage4(e3)
        d3 = self.decoder3(c, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        d1_2 = self.upconv1(d1)
        d1_3 = self.upconv2(d1_2)
        out = self.out(d1_3, d1_2, d1)
        return out
