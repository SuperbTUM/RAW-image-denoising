import torch.nn as M
from torchsummary import summary


def DepthwiseConv(in_channels, kernel_size, stride, padding):
    return M.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, groups=in_channels)


def PointwiseConv(in_channels, out_channels):
    return M.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)


class CovSepBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super().__init__()
        self.dc = DepthwiseConv(in_channels, kernel_size, stride=stride, padding=padding)
        self.pc = PointwiseConv(in_channels, out_channels)

    def forward(self, x):
        x = self.dc(x)
        x = self.pc(x)
        return x


class Encoder(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels // 4)
        self.activate = M.ReLU()
        self.sepconv2 = CovSepBlock(out_channels // 4, out_channels)
        self.proj = None
        if in_channels != out_channels:
            self.proj = CovSepBlock(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if self.proj:
            branch = self.proj(x)
        else:
            branch = x
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        x += branch
        return M.ReLU(x)


class Upsampling(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.upsample = M.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.upsample(x)


class Downsampling(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels=out_channels // 4, stride=2, padding=2)
        # self.activate = M.ReLU()
        self.activate = M.ReLU(inplace=True)
        self.sepconv2 = CovSepBlock(out_channels // 4, out_channels)
        self.branchconv = CovSepBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch = x
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        branch = self.branchconv(branch)
        return x + branch


class Decoder(M.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels, kernel_size=3)
        # self.activate = M.ReLU()
        self.activate = M.ReLU(inplace=True)
        self.sepconv2 = CovSepBlock(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        return x


class EncoderStage(M.Module):
    def __init__(self, in_channels, out_channels, num_encoder):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels)
        self.downsampling = Downsampling(in_channels, out_channels)
        self.num = num_encoder

    def forward(self, x):
        x = self.downsampling(x)
        for _ in range(self.num):
            x = self.encoder(x)
        return x


class DecoderStage(M.Module):
    def __init__(self, in_channels, out_channels, skip_in_channels):
        super().__init__()
        self.decoder = Decoder(in_channels, in_channels)
        self.upsampling = Upsampling(in_channels, out_channels)
        self.skipconnect = CovSepBlock(skip_in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        input, skip = x
        input = self.decoder(input)
        input = self.upsampling(input)
        skip = self.skipconnect(skip)
        return input + skip


class SimpleNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv = M.Conv2d(in_channels=4, out_channels=16, kernel_size=3)
        self.encoder_stage1 = EncoderStage(in_channels=16, out_channels=64, num_encoder=1)
        self.encoder_stage2 = EncoderStage(in_channels=64, out_channels=128, num_encoder=1)
        self.encoder_stage3 = EncoderStage(in_channels=128, out_channels=256, num_encoder=3)
        self.encoder_stage4 = EncoderStage(in_channels=256, out_channels=512, num_encoder=3)

        self.decoder_stage1 = DecoderStage(in_channels=512, skip_in_channels=256, out_channels=64)
        self.decoder_stage2 = DecoderStage(in_channels=64, skip_in_channels=128, out_channels=32)
        self.decoder_stage3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.decoder_stage4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16)
        self.output_layer = M.Sequential(*(Decoder(in_channels=16, out_channels=16),
                                           CovSepBlock(in_channels=16, out_channels=4)))

    def forward(self, img):
        pre = self.conv(img)
        first = self.encoder_stage1(pre)
        second = self.encoder_stage2(first)
        third = self.encoder_stage3(second)
        fourth = self.encoder_stage4(third)
        de_first = self.decoder_stage1((fourth, third))
        de_second = self.decoder_stage2((de_first, second))
        de_thrid = self.decoder_stage3((de_second, first))
        de_fourth = self.decoder_stage4((de_thrid, pre))
        output = self.output_layer(de_fourth)
        return output + img


def check():
    model = SimpleNet()
    for p in model.named_parameters():
        print(p)
    print(summary(model))
