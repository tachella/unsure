import torch
from torch import nn
from deepinv.models.drunet import test_pad
import torch.nn.functional as F
from deepinv.models.unet import BFBatchNorm2d
from deepinv.physics.blur import gaussian_blur
from deepinv.physics.functional import conv2d


class ResidualConnection(nn.Module):
    """ Residual connection """

    def __init__(self, mode='affine'):
        super().__init__()

        self.mode = mode
        if mode == 'affine':
            self.alpha = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, x, y):
        if self.mode == 'affine':
            return self.alpha * x + (1 - self.alpha) * y
        return x + y


class AffineConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mode='affine', bias=False, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode="circular", blind=True):
        if mode == 'affine':
            bias = False
        super().__init__(in_channels, out_channels, kernel_size, bias=bias,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, padding_mode=padding_mode)
        self.blind = blind
        self.mode = mode

    def affine(self, w):
        """ returns new kernels that encode affine combinations """
        return w.view(self.out_channels, -1).roll(1, 1).view(w.size()) - w + 1 / w[0, ...].numel()

    def forward(self, x):
        if self.mode != 'affine':
            return super().forward(x)
        else:
            kernel = self.affine(self.weight) if self.blind else torch.cat(
                (self.affine(self.weight[:, :-1, :, :]), self.weight[:, -1:, :, :]), dim=1)
            padding = tuple(elt for elt in reversed(self.padding) for _ in
                            range(2))  # used to translate padding arg used by Conv module to the ones used by F.pad
            padding_mode = self.padding_mode if self.padding_mode != 'zeros' else 'constant'  # used to translate padding_mode arg used by Conv module to the ones used by F.pad
            return F.conv2d(F.pad(x, padding, mode=padding_mode), kernel, stride=self.stride, dilation=self.dilation,
                            groups=self.groups)


class SortPool(nn.Module):
    """ Channel-wise sort pooling, C must be an even number """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.size()
        x1, x2 = torch.split(x.view(N, C // 2, 2, H, W), 1, dim=2)
        diff = F.relu(x1 - x2, inplace=True)
        return torch.cat((x1 - diff, x2 + diff), dim=2).view(N, C, H, W)


class EquivMaxPool(nn.Module):
    r"""
    Max pooling layer that is equivariant to translations.

    :param int kernel_size: size of the pooling window.
    :param int stride: stride of the pooling operation.
    :param int padding: padding to apply before pooling.
    :param bool circular_padding: circular padding for the convolutional layers.
    """

    def __init__(self, antialias=True, factor=2, device='cuda'):
        super(EquivMaxPool, self).__init__()
        self.antialias = antialias
        if antialias:
            self.antialias_kernel = gaussian_blur(factor/3.14).to(device)

    def downscale(self, x):
        r"""
        Apply the equivariant max pooling.

        :param torch.Tensor x: input tensor.
        """

        if self.antialias:
            x = conv2d(x, self.antialias_kernel, padding='circular')

        x1 = x[:, :, ::2, ::2].unsqueeze(0)
        x2 = x[:, :, ::2, 1::2].unsqueeze(0)
        x3 = x[:, :, 1::2, ::2].unsqueeze(0)
        x4 = x[:, :, 1::2, 1::2].unsqueeze(0)

        out = torch.cat([x1, x2, x3, x4], dim=0) # (4, B, C, H/2, W/2)

        ind = torch.norm(out, dim=(2, 3, 4), p=2) # (4, B)
        ind = torch.argmax(ind, dim=0).unsqueeze(0) # (1, B)
        d = torch.arange(4, device=x.device).unsqueeze(1).repeat(1, x.size(0)) #(4, B)
        self.ind = (d == ind).flatten()  # (4, B) -> (4*B)
        # index out with ind
        out = out.reshape(4*x.size(0), -1)
        out = out[self.ind, :].reshape(x.size(0), x.size(1), x.size(2)//2, x.size(3)//2)
        return out

    def upscale(self, x):
        B, C, H, W = x.shape

        out = torch.zeros((4, B, C, H*2, W*2), device=x.device)
        out[0, :, :, ::2, ::2] = x
        out[1, :, :, ::2, 1::2] = x
        out[2, :, :, 1::2, ::2] = x
        out[3, :, :, 1::2, 1::2] = x
        out = out.reshape(4 * B, -1)
        out = out[self.ind, :].reshape(B, C, H*2, W*2)

        if self.antialias:
            out = conv2d(out, self.antialias_kernel, padding='circular')

        return out

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='affine', bias=False, ksize=7,
                 padding_mode='circular', batch_norm=False):
        super().__init__()

        self.conv1 = AffineConv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels,
                               stride=1, padding=ksize // 2, bias=bias, padding_mode=padding_mode)
        if batch_norm:
            self.BatchNorm = BFBatchNorm2d(in_channels, use_bias=bias) if bias else nn.BatchNorm2d(in_channels)
        else:
            self.BatchNorm = nn.Identity()

        self.conv2 = AffineConv2d(in_channels, 4*in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
                               padding_mode=padding_mode)

        self.nonlin = SortPool() if mode == 'affine' else nn.ReLU(inplace=True)
        self.conv3 = AffineConv2d(4*in_channels, in_channels, kernel_size=1, stride=1, padding=0,
                               bias=bias, padding_mode=padding_mode)
        self.residual = ResidualConnection(mode)
        if in_channels != out_channels:
            self.convout = AffineConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.convout = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.BatchNorm(out)
        out = self.nonlin(out)
        out = self.conv3(out)
        out = self.residual(out, x)
        out = self.convout(out)
        return out


def ConvBlock(in_channels, out_channels, mode='affine', bias=False):
    ksize = 3
    if mode == 'affine':
        bias = False
    return nn.Sequential(
        AffineConv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            mode=mode,
            bias=bias,
            padding_mode="circular",
            groups=1,
        ),
        SortPool() if mode == 'affine' else nn.ReLU(inplace=True),
        AffineConv2d(
            out_channels, out_channels, kernel_size=ksize, stride=1, padding=ksize // 2, bias=bias, padding_mode="circular"
        ),
        SortPool() if mode == 'affine' else nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        residual=True,
        cat=True,
        mode='',
        bias=False,
        scales=4,
    ):
        super(UNet, self).__init__()
        self.name = "unet"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = residual
        self.cat = cat
        self.scales = scales

        self.hidden_channels = 64

        main_block = ConvBlock

        out_ch = self.hidden_channels

        self.ConvIn = ConvBlock(in_channels=in_channels, out_channels=out_ch, bias=bias, mode=mode)

        for i in range(1, scales):
            in_ch = out_ch
            out_ch = in_ch*2
            setattr(self, f"Downsample{i}", EquivMaxPool())
            setattr(self, f"DownBlock{i}", main_block(in_channels=in_ch, out_channels=out_ch,
                                                      bias=bias, mode=mode))

        for i in range(scales-1, 0, -1):
            in_ch = out_ch
            out_ch = in_ch // 2
            setattr(self, f"UpBlock{i}", main_block(in_channels=in_ch,
                                                       out_channels=out_ch, bias=bias, mode=mode))
            setattr(self, f"CatBlock{i}", ConvBlock(in_channels=in_ch,
                                                    out_channels=out_ch, bias=bias, mode=mode))

        self.ConvOut = AffineConv2d(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            bias=bias,
            mode=mode,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.Residual = ResidualConnection(mode=mode)


    def forward(self, x, sigma=None):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image.
        :param float sigma: noise level (not used).
        """

        factor = 2 ** (self.scales - 1)
        if x.size(2) % factor == 0 and x.size(3) % factor == 0:
            return self._forward(x)
        else:
            return test_pad(self._forward, x, modulo=factor)

    def _forward(self, x):
        cat_list = []
        m = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        x = x - m
        out = self.ConvIn(x)
        cat_list.append(out)
        for i in range(1, self.scales):
            out = getattr(self, f"Downsample{i}").downscale(out)
            out = getattr(self, f"DownBlock{i}")(out)
            if self.cat and i < self.scales - 1:  # save
                cat_list.append(out)

        for i in range(self.scales-1, 0, -1):
            out = getattr(self, f"Downsample{i}").upscale(out)
            out = getattr(self, f"UpBlock{i}")(out)
            if self.cat:
                out = torch.cat((cat_list.pop(), out), dim=1)
                out = getattr(self, f"CatBlock{i}")(out)

        out = self.ConvOut(out)
        out = self.Residual(out, x) if self.residual and self.in_channels == self.out_channels else out

        out = out + m
        return out
