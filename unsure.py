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
