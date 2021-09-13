import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init

class conv_dw(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
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
    def __init__(self, inp, oup):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_bn1x1(nn.Module):
    def __init__(self, inp, oup):
        super(conv_bn1x1, self).__init__()
        self.conv = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ClassHead(nn.Module):
    def __init__(self, inchannels=256, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, inchannels=256, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=256, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)

class FPN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FPN,self).__init__()
        self.out1 = conv_bn1x1(inp=in_channels[1], oup=in_channels[0])
        self.out2 = conv_bn1x1(inp=in_channels[2], oup=in_channels[1])
        self.out3 = conv_bn1x1(inp=in_channels[3], oup=in_channels[2])

        self.merge1 = conv_bn(inp=out_channels[0], oup=out_channels[0])
        self.merge2 = conv_bn(inp=out_channels[1], oup=out_channels[1])
        self.merge3 = conv_bn(inp=out_channels[2], oup=out_channels[2])
        self.merge4 = conv_bn(inp=out_channels[3], oup=out_channels[3])
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

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        self.conv1_norelu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv2_norelu = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv3_norelu = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )

    def forward(self, input):
        conv1_no = self.conv1_norelu(input)

        conv2_out = self.conv2(input)
        conv2_no = self.conv2_norelu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_no = self.conv3_norelu(conv3_out)

        concat_out = torch.cat([conv1_no, conv2_no, conv3_no], dim=1)
        out = F.relu(concat_out)
        return out

class FaceDetectSlimNet(nn.Module):
    def __init__(self, cfg):
        super(FaceDetectSlimNet, self).__init__()
        self.anchor = cfg["anchors"]
        self.outc = cfg["out_channels"]
        self.in_channel_list = cfg["fpn_in_list"]
        self.out_channel_list = cfg["fpn_out_list"]
        self.ssh_out_channel = cfg["ssh_out_channel"]
        # self.conv1 = conv_dw(3, 16, 2)
        # self.conv2 = conv_dw(16, 16, 1)
        # self.conv3 = conv_dw(16, 24, 2)
        # self.conv4 = conv_dw(24, 24, 1)
        # self.conv5 = conv_dw(24, 32, 2)
        # self.conv6 = conv_dw(32, 32, 1)#b 32 40 40
        #
        # self.conv7 = conv_dw(32, 48, 2)
        # self.conv8 = conv_dw(48, 48, 1)
        # self.conv9 = conv_dw(48, 48, 1)#b 64 20 20
        #
        # self.conv10 = conv_dw(48, 64, 2)
        # self.conv11 = conv_dw(64, 64, 1)
        # self.conv12 = conv_dw(64, 64, 1)#b 128 10 10
        #
        # self.conv13 = conv_dw(64, 128, 2)
        # self.conv14 = conv_dw(128, 128, 1)
        # self.conv15 = conv_dw(128, 128, 1)#b 256 5 5

        self.conv1 = conv_dw(3, self.outc[0], 2)
        self.conv2 = conv_dw(self.outc[0], self.outc[0], 1)
        self.conv3 = conv_dw(self.outc[0], self.outc[1], 2)
        self.conv4 = conv_dw(self.outc[1], self.outc[1], 1)
        self.conv5 = conv_dw(self.outc[1], self.outc[2], 2)
        self.conv6 = conv_dw(self.outc[2], self.outc[2], 1)  # b 32 40 40

        self.conv7 = conv_dw(self.outc[2], self.outc[3], 2)
        self.conv8 = conv_dw(self.outc[3], self.outc[3], 1)
        self.conv9 = conv_dw(self.outc[3], self.outc[3], 1)  # b 64 20 20

        self.conv10 = conv_dw(self.outc[3], self.outc[4], 2)
        self.conv11 = conv_dw(self.outc[4], self.outc[4], 1)
        self.conv12 = conv_dw(self.outc[4], self.outc[4], 1)  # b 128 10 10

        self.conv13 = conv_dw(self.outc[4], self.outc[5], 2)
        self.conv14 = conv_dw(self.outc[5], self.outc[5], 1)
        self.conv15 = conv_dw(self.outc[5], self.outc[5], 1)  # b 256 5 5
        self.fpn = FPN(self.in_channel_list, self.out_channel_list)
        self.ssh1 = SSH(self.out_channel_list[0], self.ssh_out_channel)
        self.ssh2 = SSH(self.out_channel_list[1], self.ssh_out_channel)
        self.ssh3 = SSH(self.out_channel_list[2], self.ssh_out_channel)
        self.ssh4 = SSH(self.out_channel_list[3], self.ssh_out_channel)
        self.ClassHead = self._make_class_head(self.ssh_out_channel)
        self.BboxHead = self._make_bbox_head(self.ssh_out_channel)
        self.LandmarkHead = self._make_landmark_head(self.ssh_out_channel)

    def _make_class_head(self, input):
        classhead = nn.ModuleList()
        classhead.append(ClassHead(input, len(self.anchor[0])))
        classhead.append(ClassHead(input, len(self.anchor[1])))
        classhead.append(ClassHead(input, len(self.anchor[2])))
        classhead.append(ClassHead(input, len(self.anchor[3])))
        return classhead

    def _make_bbox_head(self, input):
        bboxhead = nn.ModuleList()
        bboxhead.append(BboxHead(input, len(self.anchor[0])))
        bboxhead.append(BboxHead(input, len(self.anchor[1])))
        bboxhead.append(BboxHead(input, len(self.anchor[2])))
        bboxhead.append(BboxHead(input, len(self.anchor[3])))
        return bboxhead

    def _make_landmark_head(self, input):
        landmarkhead = nn.ModuleList()
        landmarkhead.append(LandmarkHead(input, len(self.anchor[0])))
        landmarkhead.append(LandmarkHead(input, len(self.anchor[1])))
        landmarkhead.append(LandmarkHead(input, len(self.anchor[2])))
        landmarkhead.append(LandmarkHead(input, len(self.anchor[3])))
        return landmarkhead

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x32 = x

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x64 = x

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x128 = x

        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x256 = x
        xes = [x32, x64, x128, x256]
        # FPN
        fpn = self.fpn(xes)
        # SSH
        ssh1 = self.ssh1(fpn[0])
        ssh2 = self.ssh2(fpn[1])
        ssh3 = self.ssh3(fpn[2])
        ssh4 = self.ssh4(fpn[3])

        features = [ssh1, ssh2, ssh3, ssh4]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # output = (bbox_regressions, classifications, ldm_regressions)#train
        output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)  # test
        return output

if __name__ == "__main__":
    import time
    from config import cfg_slimNet
    net = FaceDetectSlimNet(cfg=cfg_slimNet)
    net.eval()
    torch.save(net.state_dict(), 'facenet.pth')
    x = torch.randn(1, 3, 320, 320)
    load_t0 = time.time()
    y = net(x)
    load_t1 = time.time()
    forward_time = load_t1 - load_t0
    print("前向传播时间:{:.4f}秒".format(forward_time))
    print(y[0].size())