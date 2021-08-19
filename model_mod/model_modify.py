import torch.nn as M
from torchsummary import summary
import torch as meg


def DepthwiseConv(in_channels, kernel_size, stride, padding):
    return M.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, groups=in_channels, bias=False)


def PointwiseConv(in_channels, out_channels):
    return M.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)


class CovSepBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.dc = DepthwiseConv(in_channels, kernel_size, stride=stride, padding=padding)
        self.pc = PointwiseConv(in_channels, out_channels)

    def forward(self, x):
        x = self.dc(x)
        x = self.pc(x)
        return x


class SEBlock(M.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        self.pooling = M.AdaptiveAvgPool2d(1)
        self.fc1 = M.Linear(in_channels, in_channels//16)
        self.fc2 = M.Linear(in_channels//16, in_channels)
        self.sigmoid = M.Sigmoid()

    def forward(self, x):
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        x = self.sigmoid(x)
        return x


class Encoder(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ordinaryConv1 = CovSepBlock(in_channels=in_channels, out_channels=out_channels // 4)
        self.activate = M.ReLU(inplace=True)
        self.ordinaryConv2 = CovSepBlock(in_channels=out_channels // 4, out_channels=out_channels)
        self.scale = SEBlock(in_channels=out_channels)
        self.proj = M.Identity()
        self.activate2 = M.ReLU(inplace=True)

    def forward(self, x):
        branch = self.proj(x)
        x = self.ordinaryConv1(x)
        x = self.activate(x)
        x = self.ordinaryConv2(x)
        scale = self.scale(x)
        x = scale * x + branch
        x = self.activate2(x)
        return x


class DownSample(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ordinaryConv1 = CovSepBlock(in_channels=in_channels, out_channels=out_channels // 4, stride=2)
        self.activate = M.ReLU(inplace=True)
        self.ordinaryConv2 = CovSepBlock(in_channels=out_channels // 4, out_channels=out_channels)
        self.skipconnect = CovSepBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1)
        self.activate2 = M.ReLU(inplace=True)

    def forward(self, x):
        branch = x
        x = self.ordinaryConv1(x)
        x = self.activate(x)
        x = self.ordinaryConv2(x)
        branch = self.skipconnect(branch)
        x += branch
        x = self.activate2(x)
        return x


class Decoder(M.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.ordinaryConv1 = CovSepBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                         padding=1)
        self.activate = M.ReLU(inplace=True)
        self.ordinaryConv2 = CovSepBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                         padding=1)
        self.scale = SEBlock(in_channels=out_channels)

    def forward(self, x):
        branch = x
        x = self.ordinaryConv1(x)
        x = self.activate(x)
        x = self.ordinaryConv2(x)
        scale = self.scale(x)
        return x * scale + branch


class Upsample(M.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = M.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ordinaryConv = M.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.ordinaryConv(x)
        return x


def EncoderStage(in_channels, out_channels, num_encoder):
    seq = [
        DownSample(in_channels, out_channels),
    ]
    for _ in range(num_encoder):
        seq.append(
            Encoder(out_channels, out_channels)
        )
    return M.Sequential(*seq)


class DecoderStage(M.Module):
    def __init__(self, in_channels, out_channels, skip_in_channels):
        super().__init__()
        self.decoder = Decoder(in_channels, in_channels)
        self.upsampling = Upsample(in_channels, out_channels)
        self.skipconnect = CovSepBlock(skip_in_channels, out_channels, kernel_size=3, padding=1)
        self.activate = M.ReLU(inplace=True)

    def forward(self, x):
        input, skip = x
        input = self.decoder(input)
        input = self.upsampling(input)
        skip = self.skipconnect(skip)
        skip = self.activate(skip)
        return input + skip


class AnotherNet(M.Module):
    def __init__(self):
        super(AnotherNet, self).__init__()
        self.conv = M.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu = M.ReLU(inplace=True)
        self.encoder_stage1 = EncoderStage(in_channels=16, out_channels=64, num_encoder=1)
        self.encoder_stage2 = EncoderStage(in_channels=64, out_channels=128, num_encoder=1)
        self.encoder_stage3 = EncoderStage(in_channels=128, out_channels=256, num_encoder=3)
        self.encoder_stage4 = EncoderStage(in_channels=256, out_channels=512, num_encoder=3)

        # Strange ???
        self.enc2dec = CovSepBlock(in_channels=512, out_channels=64, kernel_size=3, padding=1)
        self.med_activate = M.ReLU(inplace=True)

        self.decoder_stage1 = DecoderStage(in_channels=64, skip_in_channels=256, out_channels=64)
        self.decoder_stage2 = DecoderStage(in_channels=64, skip_in_channels=128, out_channels=32)
        self.decoder_stage3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.decoder_stage4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16)
        self.output_layer = M.Sequential(*(Decoder(in_channels=16, out_channels=16),
                                           M.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)))

    def forward(self, img):
        assert img.shape[1] == 4
        pre = self.conv(img)
        pre = self.relu(pre)
        first = self.encoder_stage1(pre)
        second = self.encoder_stage2(first)
        third = self.encoder_stage3(second)
        fourth = self.encoder_stage4(third)

        med = self.enc2dec(fourth)
        med = self.med_activate(med)

        de_first = self.decoder_stage1((med, third))
        de_second = self.decoder_stage2((de_first, second))
        de_thrid = self.decoder_stage3((de_second, first))
        de_fourth = self.decoder_stage4((de_thrid, pre))
        output = self.output_layer(de_fourth)
        return output + img


def check():
    model = AnotherNet()
    # for p in model.named_parameters():
    #     print(p)
    print(summary(model))


if __name__ == '__main__':
    a = meg.ones((4, 3))
    a = a.unsqueeze(2)
    a = a.unsqueeze(3)
    print(a.shape)
