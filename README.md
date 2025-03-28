# UNSURE: self-supervised learning with Unknown Noise level and Stein's Unbiased Risk Estimate

Code of the **ICLR 2025** paper "UNSURE: Unknown Noise level Stein's Unbiased Risk Estimator" by [Julian Tachella](https://tachella.github.io/),
[Mike Davies](https://www.eng.ed.ac.uk/about/people/professor-michael-e-davies) and [Laurent Jacques](https://laurentjacques.gitlab.io/).

We use the [deepinv library](https://deepinv.github.io/deepinv/) for most of the code. 

The UNSURE loss was added to the deepinv library, please see [this jupyter notebook demo](https://deepinv.github.io/deepinv/auto_examples/self-supervised-learning/demo_unsure.html) and [the documentation](https://deepinv.github.io/deepinv/api/stubs/deepinv.loss.SureGaussianLoss.html#deepinv.loss.SureGaussianLoss).

Paper available at [openreview](https://openreview.net/forum?id=ScVnYBaSEw).

# Method Description
UNSURE is a self-supervised learning loss that can be used for learning a reconstruction network $f$ 
from a dataset of noisy measurements

$$
y_i = \mathcal{S}(Ax_i) 
$$

for $i=1,\dots,N$ where $x_i$ is the clean image, $y_i$ is the noisy measurement, $A$ is a linear operator and $\mathcal{S}$ is a stochastic noising process.

Unlike Stein's Unbiased Risk Estimator (SURE), the proposed loss can be used without any prior knowledge of the noise level. 
The loss is defined as

$$
\max_{\eta} \min_{f} \sum_{i=1}^N ||y_i - f(y_i)||^2 + 2\Sigma_{\eta} \text{div} f(y_i) 
$$

where $\eta$ is a Lagrange multiplier, 
$\Sigma_{\eta}$ is the covariance matrix of the noise and $\text{div} f(y)$ is the divergence of the network $f$ at the point $y$.
There exist multiple variants for different types of noise (e.g. Gaussian, Poisson-Gaussian, spatially correlated noise, etc.). 
Please refer to the paper for more details.

# Getting Started
1. Clone the repository
2. Install the latest version of [deepinv](https://deepinv.github.io/) if you don't have it already
```
pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
```
3. Generate the datasets by running the `generate_datasets.py` file.
4. Run the `main.py` file.

# Citation
```
@inproceedings{
tachella2025unsure,
title={{UNSURE}: self-supervised learning with Unknown Noise level  and Stein's Unbiased Risk Estimate},
author={Juli{\'a}n Tachella and Mike Davies and Laurent Jacques},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=ScVnYBaSEw}
}
```
