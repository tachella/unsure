from torchvision import transforms
import torchvision
import deepinv as dinv
import torch
from deepinv.utils.demo import load_dataset, load_degradation
from pathlib import Path
import os
from PIL import Image
from torch.utils.data import Dataset
from models import UNet
from unsure import UNSURE


def get_unrolled(channels, norm, device, algo='PGD', scales=3, weight_tied=True, iter=3):
    unrolled_iter = iter
    if weight_tied:
        prior = dinv.optim.PnP(denoiser=UNet(in_channels=channels, out_channels=channels, scales=scales))
    else:
        prior = [dinv.optim.PnP(denoiser=UNet(in_channels=channels, out_channels=channels, scales=scales)) for _ in range(unrolled_iter)]
    return dinv.unfolded.unfolded_builder(
        algo,
        params_algo={"stepsize": [norm] * unrolled_iter, "g_param": [0.01] * unrolled_iter, "lambda": 1.0},
        trainable_params=["lambda", "stepsize", "g_param"],
        data_fidelity=dinv.optim.L2(),
        max_iter=unrolled_iter,
        prior=prior,
        verbose=False,
    ).to(device)


def get_problem(problem, generate_dataset=False):
    device = dinv.utils.get_freer_gpu()
    if problem == 'Tomography':
        pix = 128
        sigma = .0005
        batch_size = 8
        physics = dinv.physics.Tomography(angles=pix, img_width=pix, normalize=True, parallel_computation=True,
                                          device=device)
        physics.noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=sigma)
        path = './Tomography/'

        transform = transforms.Compose([transforms.Resize(pix)])
        if generate_dataset:
            p = "./Tomography/"
            data_train = dinv.datasets.HDF5Dataset(p + 'dinv_dataset0.h5', train=True, transform=transform)
            data_test = dinv.datasets.HDF5Dataset(p + 'dinv_dataset0.h5', train=False, transform=transform)
        else:
            data_train = [dinv.datasets.HDF5Dataset(path + 'dinv_dataset0.h5', train=True)]
            data_test = [dinv.datasets.HDF5Dataset(path + 'dinv_dataset0.h5', train=False)]

        norm = physics.compute_norm(torch.ones(1, 1, pix, pix, device=device))

        channels = 1
        model = get_unrolled(channels, norm, device, algo='PGD', scales=2, iter=4, weight_tied=False)
    elif problem == 'MRI':

        batch_size = 8
        sigma = .03
        train_dataset_name = "fastmri_knee_singlecoil"
        img_size = 128
        path = './MRI/'

        transform = transforms.Compose([transforms.Resize(img_size)])

        if generate_dataset:
            data_train = load_dataset(
                train_dataset_name, Path(path), transform, train=True
            )
            data_test = load_dataset(
                train_dataset_name, Path(path), transform, train=False
            )
        else:
            data_train = [dinv.datasets.HDF5Dataset(path=path+'dinv_dataset0.h5', train=True)]
            data_test = [dinv.datasets.HDF5Dataset(path=path+'dinv_dataset0.h5', train=False)]
        norm = 1.

        mask = load_degradation("mri_mask_128x128.npy", Path(path))
        physics = dinv.physics.MRI(mask=mask, device=device, noise_model=dinv.physics.GaussianNoise(sigma=sigma))
        channels = 2

        model = get_unrolled(channels, norm, device, algo='HQS', scales=2, iter=7, weight_tied=False)

    elif problem.startswith('MNIST_denoising'):
        index = int(problem[-1])
        sigma = [.05, .1, .2, .3, .4, .5][index]
        batch_size = 256
        path = f'./MNIST_denoising{index}/'

        transform = transforms.Compose([transforms.ToTensor()])

        if generate_dataset:
            data_train = torchvision.datasets.MNIST(root=path, train=True, transform=transform, download=True)
            data_test = torchvision.datasets.MNIST(root=path, train=False, transform=transform)
        else:
            data_train = [dinv.datasets.HDF5Dataset(path=path+'dinv_dataset0.h5', train=True)]
            data_test = [dinv.datasets.HDF5Dataset(path=path+'dinv_dataset0.h5', train=False)]

        physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=sigma))
        channels = 1

        model = UNet(in_channels=channels, out_channels=channels, scales=3, bias=False).to(device)
        model = dinv.models.ArtifactRemoval(model)

    elif problem == 'Denoising_div2k':

        sigma = .2
        batch_size = 8
        img_size = 128
        path = './denoising/'

        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size))])

        physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(sigma=sigma))
        if generate_dataset:
            data_train = dinv.datasets.DIV2K(root='./demosaicing/', mode='train', transform=transform, download=True)
            data_test = dinv.datasets.DIV2K(root='./demosaicing/', mode='val', transform=transform)
        else:
            data_train = [dinv.datasets.HDF5Dataset(path=path+f'dinv_dataset0.h5', train=True)]
            data_test = [dinv.datasets.HDF5Dataset(path=path+f'dinv_dataset0.h5', train=False)]

        model = dinv.models.DRUNet()
        model = dinv.models.ArtifactRemoval(model).to(device)

    elif problem == 'DenoisingCorrelated_div2k':
        sigma = .2
        batch_size = 14
        img_size = 256
        path = './denoising_corr/'

        filter = torch.ones((1, 1, 3, 3), device=device)/9
        transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((img_size, img_size))])
        op = lambda x: dinv.physics.functional.conv2d(x, filter, padding='circular')
        physics = dinv.physics.Denoising(noise_model=CorrelatedGaussianNoise(sigma=sigma, linear_op=op))
        if generate_dataset:
            data_train = dinv.datasets.DIV2K(root='./demosaicing/', mode='train', transform=transform, download=True)
            data_test = dinv.datasets.DIV2K(root='./demosaicing/', mode='val', transform=transform)
        else:
            data_train = [dinv.datasets.HDF5Dataset(path=path+f'dinv_dataset0.h5', train=True)]
            data_test = [dinv.datasets.HDF5Dataset(path=path+f'dinv_dataset0.h5', train=False)]

        channels = 3
        model = UNet(in_channels=channels, out_channels=channels, scales=4, bias=False).to(device)
        model = dinv.models.ArtifactRemoval(model)

    elif problem == 'DenoisingPoissonGaussian_div2k':

        sigma = .1
        batch_size = 16
        img_size = 128
        path = './denoising_poisson_gaussian/'

        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size))])

        physics = dinv.physics.Denoising(noise_model=dinv.physics.PoissonNoise(gain=sigma))
        if generate_dataset:
            data_train = dinv.datasets.DIV2K(root='./demosaicing/', mode='train', transform=transform, download=True)
            data_test = dinv.datasets.DIV2K(root='./demosaicing/', mode='val', transform=transform)
        else:
            data_train = [dinv.datasets.HDF5Dataset(path=path+f'dinv_dataset0.h5', train=True)]
            data_test = [dinv.datasets.HDF5Dataset(path=path+f'dinv_dataset0.h5', train=False)]

        channels = 3
        model = UNet(in_channels=channels, out_channels=channels, scales=5, bias=False).to(device)
        model = dinv.models.ArtifactRemoval(model)
    else:
        raise ValueError('Problem not recognized')

    return model, physics, data_train, data_test, batch_size, device, path


def get_losses(method, model, sigma, physics, problem=None, device='cuda'):
    if method == 'unsure':
        losses = [UNSURE(device=device)]
    elif method == 'unsure_score':
        losses = [dinv.loss.ScoreLoss()]
        model = losses[0].adapt_model(model)
    elif method == 'unsure_corr':
        losses = [UNSURE(kernel_size=3, device=device), dinv.loss.MCLoss()]
    elif method == 'unsure_poisson_gaussian':
        s = 0.0005
        losses = [UNSURE(mode='poisson_gaussian', sigma_init=s, gain_init=s,
                                  step_size=s/100, device=device)]
    elif method == 'mc':
        losses = [dinv.loss.MCLoss()]
    elif method == 'sup':
        losses = [dinv.loss.SupLoss()]
    elif method.startswith('splitting'):
        if problem == 'MRIs':
            generator = dinv.physics.generator.RandomMaskGenerator(img_size=(2, 128, 128), center_fraction=0.01,
                                                                   acceleration=1.5, device=device)
        else:
            generator = None

        losses = [dinv.loss.SplittingLoss(split_ratio=.8, MC_samples=5,
                                          mask_generator=generator)]
        model = losses[0].adapt_model(model)
    elif method == 'r2r':
        losses = [dinv.loss.R2RLoss(sigma=sigma)]
        model = losses[0].adapt_model(model, MC_samples=5)
    elif method == 'sure':
        if problem == 'DenoisingPoisson_div2k':
            losses = [dinv.loss.SurePoissonLoss(gain=sigma)]
        elif problem == 'tomography':
            losses = [dinv.loss.SurePGLoss(gain=sigma, sigma=sigma)]
        else:
            losses = [dinv.loss.SureGaussianLoss(sigma=sigma)]
    elif method == 'neigh2neigh':
        losses = [dinv.loss.Neighbor2Neighbor()]
    else:
        raise NotImplementedError
    if problem == 'MRI' and method != 'sup':
        losses.append(dinv.loss.EILoss(dinv.transform.Rotate()))
    elif problem == 'Demosaicing_div2k':
        if "noEI" not in method and method != 'sup':
            losses.append(dinv.loss.EILoss(dinv.transform.Shift(), weight=.5))

    return model, losses


class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.image_paths = [os.path.join(root, filename) for filename
                            in os.listdir(root) if os.path.isfile(os.path.join(root, filename))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image


class CorrelatedGaussianNoise(torch.nn.Module):
    r"""

    Gaussian noise :math:`y=z+\epsilon` where :math:`\epsilon\sim \mathcal{N}(0,I\sigma^2)`.

    |sep|

    :Examples:

        Adding gaussian noise to a physics operator by setting the ``noise_model``
        attribute of the physics operator:

        >>> from deepinv.physics import Denoising, GaussianNoise
        >>> import torch
        >>> physics = Denoising()
        >>> physics.noise_model = GaussianNoise()
        >>> x = torch.rand(1, 1, 2, 2)
        >>> y = physics(x)

    :param float sigma: Standard deviation of the noise.

    """

    def __init__(self, sigma=0.1, linear_op=lambda x: x):
        super().__init__()
        self.update_parameters(sigma)
        self.linear_op = linear_op

    def forward(self, x, sigma=None, **kwargs):
        r"""
        Adds the noise to measurements x

        :param torch.Tensor x: measurements
        :param float, torch.Tensor sigma: standard deviation of the noise.
            If not None, it will overwrite the current noise level.
        :returns: noisy measurements
        """
        self.update_parameters(sigma)
        return x + self.linear_op(torch.randn_like(x) * self.sigma)

    def update_parameters(self, sigma=None, **kwargs):
        r"""
        Updates the standard deviation of the noise.

        :param float, torch.Tensor sigma: standard deviation of the noise.
        """
        if sigma is not None:
            self.sigma = sigma
