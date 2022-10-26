import torch
import torch.nn as nn
import utils
import numpy as np

pi = np.pi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#siganl phase-only layer
class PhaseLayer(nn.Module):
    def __init__(self, row: int, col: int) ->None:
        super(PhaseLayer, self).__init__()
        #row and col are layer size(pixel)
        #PhaseC is phase constrain such as 2pi or pi
        self.row = row
        self.col = col
        self.weight = nn.Parameter(torch.randn(row, col))

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        phase = torch.sigmoid(self.weight)  #tanh
        # phase = (phase>=0)*pi
        tem = torch.exp(1j*2*pi*phase)
        return input*tem


# class Discriminator(nn.Module):
#     def __init__(self,
#                  real_size,
#                  in_distance,
#                  layer_interval,
#                  out_distance,
#                  wave_length,
#                  row: int,
#                  col: int
#                  ):
#         super(Discriminator, self).__init__()
#         # init parameter
#         self.real_size = real_size
#         self.wave_length = wave_length
#         self.row = row
#         self.col = col
#
#         # instance layer
#         self.in_distance = in_distance
#         self.out_distance = out_distance
#         self.layer_interval = layer_interval
#         self.model1 = PhaseLayer(row, col)
#         self.model2 = PhaseLayer(row, col)
#         # self.model3 = PhaseLayer(row, col)
#
#     def forward(self, x):
#         [bat, cha, row, col] = x.shape
#
#         x = utils.transmission_by_specturm(self.real_size, x, self.in_distance, self.wave_length, device=device)
#
#         x = self.model1(x)
#         x = utils.transmission_by_specturm(self.real_size, x, self.layer_interval, self.wave_length, device=device)
#
#         # x = self.model2(x)
#         # x = utils.transmission_by_specturm(self.real_size, x, self.layer_interval, self.wave_length, device=device)
#
#         x = self.model2(x)
#         x = utils.transmission_by_specturm(self.real_size, x, self.out_distance, self.wave_length, device=device)
#
#         x = torch.abs(x)**2
#         x = utils.area_energy(x, device=device)
#
#         return x


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            #Input: N x channels_img x 128 x 128
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),# 64x64
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1), #32x32
            self._block(features_d*2, features_d * 4, 4, 2, 1), #16x16
            self._block(features_d*4, features_d * 8, 4, 2, 1), #8x8
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0), #1x1

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = False,
            ),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self,
                 real_size,
                 in_distance,
                 layer_interval,
                 out_distance,
                 wave_length,
                 row: int,
                 col: int
                 ):
        super(Generator, self).__init__()
        # init parameter
        self.real_size = real_size
        self.wave_length = wave_length
        self.row = row
        self.col = col

        # instance layer
        self.in_distance = in_distance
        self.out_distance = out_distance
        self.layer_interval = layer_interval
        self.model1 = PhaseLayer(row, col)
        self.model2 = PhaseLayer(row, col)
        self.model3 = PhaseLayer(row, col)

    def forward(self, x):
        [bat, cha, row, col] = x.shape

        x = utils.transmission_by_specturm(self.real_size, x, self.in_distance, self.wave_length, device=device)

        x = self.model1(x)
        x = utils.transmission_by_specturm(self.real_size, x, self.layer_interval, self.wave_length, device=device)

        x = self.model2(x)
        x = utils.transmission_by_specturm(self.real_size, x, self.layer_interval, self.wave_length, device=device)

        x = self.model3(x)
        x = utils.transmission_by_specturm(self.real_size, x, self.out_distance, self.wave_length, device=device)

        x = torch.abs(x)**2

        return x

# class Generator(nn.Module):
#     def __init__(self, z_dim, channels_img, features_g, row: int,
#                  col: int):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             #Input: N x z_dim x 1 x 1
#             self._block(z_dim, features_g*16, 4, 1, 0), #N x f_g*16 x 4 x 4
#             self._block(features_g*16, features_g*8, 4, 2, 1),  #8x8
#             self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
#             self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
#             self._block(features_g * 2, features_g, 4, 2, 1),  #64x64
#             nn.ConvTranspose2d(
#                 features_g, channels_img, kernel_size=4, stride=2, padding=1,
#             ), #128x128
#             nn.ConstantPad2d((int((col-128)/2), int((col-128)/2), int((row-128)/2), int((row-128)/2)), 0)
#         )
#
#     def _block(self, in_channels, out_channels, kernel_size, stride, padding):
#         return nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         return self.gen(x)



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, PhaseLayer)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    lam = 632.8 * 1e-9
    pix_size = 12.5 * 1e-6
    M = 256
    all_size = pix_size * M
    layer_interval = 20 * 1e-3
    in_distance = 30 * 1e-3
    out_distance = 30 * 1e-3
    N, in_channels, H, W = 8, 1, 256, 256
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(all_size, in_distance, layer_interval,
                         out_distance, lam, M, M)
    initialize_weights(disc)
    mm = disc(x).shape
    assert disc(x).shape == (N, 10)
    gen = Generator(z_dim, in_channels, 8, H, W)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    m = gen(z).shape
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")


# test()