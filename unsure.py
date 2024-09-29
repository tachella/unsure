import deepinv as dinv
import torch

class UNSURE(dinv.loss.Loss):
    def __init__(self,  mode='gaussian', tau=0.05, step_size=0.01, sigma_init=.1, gain_init=.1, kernel_size=1,
                 momentum=.9, pseudo_inverse=True, device='cpu'):
        r"""
        Unknown Noise level Stein's Unbiased Risk Estimator (UNSURE) loss.

        :param str mode: 'gaussian', 'poisson' or 'poisson_gaussian'.
        :param float tau: Constant for approximating the divergence using a Monte Carlo estimate.
        :param float step_size: Gradient step size for Lagrange multipliers.
        :param float sigma_init: Initial value of the Lagrange multiplier related to the Gaussian noise.
        :param float gain_init: Initial value of the Lagrange multiplier related to the Poisson noise.
        :param int kernel_size: Size of the kernel controlling the noise spatial correlation. Default is 1 (no blur).
        :param float momentum: Momentum.
        :param bool pseudo_inverse: Correct loss using pseudo-inverse.
        :param str device: Device (cpu or gpu).
        """
        super(UNSURE, self).__init__()
        self.tau = tau
        self.kernel_size = kernel_size
        self.gain = 0.
        self.sigma = 0.

        # initialise Lagrange multipliers
        if mode == 'gaussian' or mode == 'poisson_gaussian':
            self.sigma = torch.ones((1, 1, kernel_size, kernel_size), device=device)
            self.sigma = self.sigma/self.sigma.sum()*sigma_init
            self.sigma.requires_grad = True
        if mode == 'poisson' or mode == 'poisson_gaussian':
            self.gain = torch.ones((1, 1, kernel_size, kernel_size), device=device)
            self.gain = self.gain/self.gain.sum()*gain_init
            self.gain.requires_grad = True

        self.mode = mode
        self.step_size = step_size
        self.grad_sigma = 0.
        self.grad_gain = 0.
        self.momentum = momentum
        self.init_flag = True
        self.pinv = pseudo_inverse

    def forward(self, y, x_net, physics, model, **kwargs):
        y1 = physics.A(x_net)

        b = torch.randn_like(y)

        if self.mode == 'poisson' or self.mode == 'poisson_gaussian':
            r = torch.sqrt(y)
            r[y <= 0] = 0
            gain = self.gain.sqrt()*r
        else:
            gain = 0.

        if self.kernel_size > 1:
            b = dinv.physics.functional.conv2d(b, (self.sigma+gain), padding='circular')
        else:
            b *= (self.sigma+gain)

        y2 = physics.A(model(y + b * self.tau, physics))

        if self.pinv:
            diff = physics.A_dagger(b) * physics.A_dagger(y2 - y1) / self.tau
        else:
            diff = (b * (y2 - y1)) / self.tau
        div = 2*diff.reshape(y.size(0), -1).mean(1)


        if self.mode == 'gaussian' or self.mode == 'poisson_gaussian':
            self.gradient_step_sigma(div.mean())
        if self.mode == 'poisson' or self.mode == 'poisson_gaussian':
            self.gradient_step_gain(div.mean())

        if self.pinv:
            residual = physics.A_dagger(y1-y).pow(2).reshape(y.size(0), -1).mean(1)
        else:
            residual = (y1-y).pow(2).reshape(y.size(0), -1).mean(1)

        loss = div + residual
        return loss

    def gradient_step(self, loss, param, saved_grad):
        grad = torch.autograd.grad(loss, param, retain_graph=True)[0]
        if self.init_flag:
            self.init_flag = False
            saved_grad = grad
        else:
            saved_grad = self.momentum*saved_grad + (1.-self.momentum)*grad
        return param + self.step_size*grad, saved_grad

    def gradient_step_sigma(self, loss):
        self.sigma, self.grad_sigma = self.gradient_step(loss, self.sigma, self.grad_sigma)

    def gradient_step_gain(self, loss):
        self.gain, self.grad_gain = self.gradient_step(loss, self.gain, self.grad_gain)




class ScoreLoss(dinv.loss.Loss):
    r"""
    Learns score of noise distribution.

    Approximates the score of the measurement distribution :math:`S(y)\approx \nabla \log p(y)`
    https://proceedings.neurips.cc/paper_files/paper/2021/file/077b83af57538aa183971a2fe0971ec1-Paper.pdf.

    The score loss is defined as

    .. math::

        \| \epsilon + \sigma S(y+ \sigma \epsilon) \|^2

    where :math:`y` is the noisy measurement,
    :math:`S` is the model approximating the score of the noisy measurement distribution :math:`\nabla \log p(y)`,
    :math:`\epsilon` is sampled from :math:`N(0,I)` and
    :math:`\sigma` is sampled from :math:`N(0,I\delta^2)` with :math:`\delta` annealed during training
    from a maximum value to a minimum value.

    At test/evaluation time, the method uses Tweedie's formula to estimate the score,
    which depends on the noise model used:

    - UNSURE: :math:`R(y) = y + \frac{n}{\|S(y)\|^2} S(y)`
    - Gaussian noise: :math:`R(y) = y + \sigma^2 S(y)`
    - Poisson noise: :math:`R(y) = y + \gamma y S(y)`
    - Gamma noise: :math:`R(y) = \frac{\ell y}{(\ell-1)-y S(y)}`

    .. warning::

        The user should provide a backbone model :math:`S`
        to :meth:`adapt_model` which returns the full reconstruction network
        :math:`R`, which is mandatory to compute the loss properly.

    .. warning::

        This class uses the inference formula for the Poisson noise case
        which differs from the one proposed in Noise2Score.

    .. note::

        This class does not support general inverse problems, it is only designed for denoising problems.

    :param None, torch.nn.Module noise_model: Noise distribution corrupting the measurements
        (see :ref:`the physics docs <physics>`). Options are :class:`deepinv.physics.GaussianNoise`,
        :class:`deepinv.physics.PoissonNoise`, :class:`deepinv.physics.GammaNoise` and
        :class:`deepinv.physics.UniformGaussianNoise`. By default, it uses the noise model associated with
        the physics operator provided in the forward method.
    :param int total_batches: Total number of training batches (epochs * number of batches per epoch).
    :param tuple delta: Tuple of two floats representing the minimum and maximum noise level,
        which are annealed during training.

    """
    def __init__(self, noise_model=None, total_batches=1000, delta=(0.001, 0.1)):
        super(ScoreLoss, self).__init__()
        self.total_batches = total_batches
        self.delta = delta
        self.noise_model = noise_model

    def forward(self, model, **kwargs):
        r"""
        Computes the Score Loss.

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction model.
        :return: (torch.Tensor) Score loss.
        """
        return model.get_error()

    def adapt_model(self, model, **kwargs):
        r"""
        Transforms score backbone net :meth:`S` into :meth:`R` for training and evaluation.

        :param torch.nn.Module model: Backbone model approximating the score.
        :return: (torch.nn.Module) Adapted reconstruction model.
        """
        if isinstance(model, ScoreModel):
            return model
        else:
            return ScoreModel(model, self.noise_model, self.delta, self.total_batches)




class ScoreModel(torch.nn.Module):
    r"""
    Score model for the ScoreLoss.


    :param torch.nn.Module model: Backbone model approximating the score.
    :param None, torch.nn.Module noise_model: Noise distribution corrupting the measurements
        (see :ref:`the physics docs <physics>`). Options are :class:`deepinv.physics.GaussianNoise`,
        :class:`deepinv.physics.PoissonNoise`, :class:`deepinv.physics.GammaNoise` and
        :class:`deepinv.physics.UniformGaussianNoise`. By default, it uses the noise model associated with
        the physics operator provided in the forward method.
    :param tuple delta: Tuple of two floats representing the minimum and maximum noise level,
        which are annealed during training.
    :param int total_batches: Total number of training batches (epochs * number of batches per epoch).

    """

    def __init__(self, model, noise_model, delta, total_batches):
        super(ScoreModel, self).__init__()
        self.base_model = model
        self.min = delta[0]
        self.max = delta[1]
        self.noise_model = noise_model
        self.counter = 0
        self.total_batches = total_batches

    def forward(self, y, physics, update_parameters=False):
        r"""
        Computes the reconstruction of the noisy measurements.

        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param bool update_parameters: If True, updates the parameters of the model.
        """

        if self.noise_model is None:
            noise_model = "unsure"
        else:
            noise_model = self.noise_model

        noise_class = noise_model.__class__.__name__

        if self.training:
            self.counter += 1
            w = self.counter / self.total_batches
            delta = self.max * (1 - w) + self.min * w
            sigma = (
                torch.randn((y.size(0),) + (1,) * (y.dim() - 1), device=y.device)
                * delta
            )
        else:
            sigma = self.min

        extra_noise = torch.randn_like(y)

        y_plus = y + extra_noise * sigma

        grad = self.base_model(y_plus, physics)

        if update_parameters:
            error = extra_noise + grad * sigma
            self.error = error.pow(2).mean()

        if noise_class in ["unsure"]:
            step = 1 / grad.pow(2).mean()
            out = y + step * grad
        elif noise_class in ["GaussianNoise", "UniformGaussianNoise"]:
            out = y + noise_model.sigma**2 * grad
        elif noise_class == "PoissonNoise":
            if not noise_model.normalize:
                y *= noise_model.gain
            out = y + noise_model.gain * y * grad
        elif noise_class == "GammaNoise":
            l = noise_model.l
            out = l * y / ((l - 1.0) - y * grad)
        else:
            raise NotImplementedError(f"Noise model {noise_class} not implemented")

        return out

    def get_error(self):
        return self.error