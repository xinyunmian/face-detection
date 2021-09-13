import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class conv_dw(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, p=1):
        super(conv_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, kernel_size=k, stride=s, padding=p, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class conv_bn(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, p=1):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPN(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(FPN,self).__init__()
        self.out1 = conv_dw(inp=in_channels[1], oup=in_channels[0], k=1, s=1, p=0)
        self.out2 = conv_dw(inp=in_channels[2], oup=in_channels[1], k=1, s=1, p=0)
        self.out3 = conv_dw(inp=in_channels[3], oup=in_channels[2], k=1, s=1, p=0)

        self.merge1 = conv_dw(inp=out_channels[0], oup=out_channels[0])
        self.merge2 = conv_dw(inp=out_channels[1], oup=out_channels[1])
        self.merge3 = conv_dw(inp=out_channels[2], oup=out_channels[2])
        self.merge4 = conv_dw(inp=out_channels[3], oup=out_channels[3])
    def forward(self, inputs):
        output11 = self.merge4(inputs[3])  # b 256 5 5 -> b 256 5 5

        output4 = self.out3(inputs[3])  # b 256 5 5 -> b 128 5 5
        up4 = F.interpolate(output4, scale_factor=2, mode="nearest")
        # up4 = F.interpolate(output4, size=[inputs[2].size(2), inputs[2].size(3)], mode="nearest")
        output44 = inputs[2] + up4
        output44 = self.merge3(output44)# b 128 10 10

        output3 = self.out2(output44)  # b 128 10 10 -> b 64 10 10
        up3 = F.interpolate(output3, scale_factor=2, mode="nearest")
        # up3 = F.interpolate(output3, size=[inputs[1].size(2), inputs[1].size(3)], mode="nearest")
        output33 = inputs[1] + up3
        output33 = self.merge2(output33)  # b 64 20 20

        output2 = self.out1(output33)  # b 64 20 20 -> b 32 20 20
        up2 = F.interpolate(output2, scale_factor=2, mode="nearest")
        # up2 = F.interpolate(output2, size=[inputs[0].size(2), inputs[0].size(3)], mode="nearest")
        output22 = inputs[0] + up2
        output22 = self.merge1(output22)  # b 32 40 40
        out = [output22, output33, output44, output11]
        return out

class FaceYoloHead(nn.Module):
    def __init__(self, feature_channels=[48, 64, 96, 128], target_channel=48):
        super(FaceYoloHead, self).__init__()

        self.conv8 = conv_dw(feature_channels[0], feature_channels[0] // 2, s=1)
        self.conv88 = conv_dw(feature_channels[0] // 2, feature_channels[0], s=1)
        self.out8 = conv_dw(feature_channels[0], target_channel, s=1)

        self.conv16 = conv_dw(feature_channels[1], feature_channels[1] // 2, s=1)
        self.conv166 = conv_dw(feature_channels[1] // 2, feature_channels[1], s=1)
        self.out16 = conv_dw(feature_channels[1], target_channel, s=1)

        self.conv32 = conv_dw(feature_channels[2], feature_channels[2] // 2, s=1)
        self.conv322 = conv_dw(feature_channels[2] // 2, feature_channels[2], s=1)
        self.out32 = conv_dw(feature_channels[2], target_channel, s=1)

        self.conv64 = conv_dw(feature_channels[3], feature_channels[3] // 2, s=1)
        self.conv644 = conv_dw(feature_channels[3] // 2, feature_channels[3], s=1)
        self.out64 = conv_dw(feature_channels[3], target_channel, s=1)

    def forward(self, features):

        feature8 = features[0]
        pre8 = self.conv8(feature8)
        pre8 = self.conv88(pre8)
        pre8 = self.out8(pre8)


        feature16 = features[1]
        pre16 = self.conv16(feature16)
        pre16 = self.conv166(pre16)
        pre16 = self.out16(pre16)


        feature32 = features[2]
        pre32 = self.conv32(feature32)
        pre32 = self.conv322(pre32)
        pre32 = self.out32(pre32)

        feature64 = features[3]
        pre64 = self.conv64(feature64)
        pre64 = self.conv644(pre64)
        pre64 = self.out64(pre64)

        return pre8, pre16, pre32, pre64

class FaceYoloNet(nn.Module):
    def __init__(self, cfg):
        super(FaceYoloNet, self).__init__()
        self.outc = cfg.out_channels
        self.targetc = cfg.target_outc
        self.in_channel_list = cfg.fpn_in_list
        self.out_channel_list = cfg.fpn_out_list

        self.conv1 = conv_dw(inp=3, oup=self.outc[0], s=2)
        self.conv2 = conv_dw(inp=self.outc[0], oup=self.outc[0], s=1)
        self.conv22 = conv_dw(inp=self.outc[0], oup=self.outc[0], s=1)
        self.conv3 = conv_dw(inp=self.outc[0], oup=self.outc[1], s=2)
        self.conv4 = conv_dw(inp=self.outc[1], oup=self.outc[1], s=1)
        self.conv44 = conv_dw(inp=self.outc[1], oup=self.outc[1], s=1)
        self.conv5 = conv_dw(inp=self.outc[1], oup=self.outc[2], s=2)
        self.conv6 = conv_dw(inp=self.outc[2], oup=self.outc[2], s=1)  # b 32 80 80
        self.conv66 = conv_dw(inp=self.outc[2], oup=self.outc[2], s=1)

        self.conv7 = conv_dw(inp=self.outc[2], oup=self.outc[3], s=2)
        self.conv8 = conv_dw(inp=self.outc[3], oup=self.outc[3], s=1)
        self.conv9 = conv_dw(inp=self.outc[3], oup=self.outc[3], s=1)  # b 64 40 40

        self.conv10 = conv_dw(inp=self.outc[3], oup=self.outc[4], s=2)
        self.conv11 = conv_dw(inp=self.outc[4], oup=self.outc[4], s=1)
        self.conv12 = conv_dw(inp=self.outc[4], oup=self.outc[4], s=1)  # b 128 20 20

        self.conv13 = conv_dw(inp=self.outc[4], oup=self.outc[5], s=2)
        self.conv14 = conv_dw(inp=self.outc[5], oup=self.outc[5], s=1)
        self.conv15 = conv_dw(inp=self.outc[5], oup=self.outc[5], s=1)  # b 256 10 10
        self.fpn = FPN(self.in_channel_list, self.out_channel_list)
        self.head = FaceYoloHead(self.out_channel_list, self.targetc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv44(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv66(x)
        x8 = x

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x16 = x

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x32 = x

        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x64 = x
        xes = [x8, x16, x32, x64]
        # FPN
        fpn = self.fpn(xes)
        out8, out16, out32, out64 = self.head(fpn)
        return out8, out16, out32, out64

if __name__ == "__main__":
    from FaceConfig import facecfg
    net = FaceYoloNet(cfg=facecfg)
    net.eval()
    torch.save(net.state_dict(), 'weights/YoloFace.pth')
    x = torch.randn(1, 3, 640, 640)
    y = net(x)
    print(y[0].size())


























