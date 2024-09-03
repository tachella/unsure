# UNSURE: Unknown Noise level Stein's Unbiased Risk Estimator

Code of the paper "UNSURE: Unknown Noise level Stein's Unbiased Risk Estimator" by [Julian Tachella](https://tachella.github.io/),
[Mike Davies](https://www.eng.ed.ac.uk/about/people/professor-michael-e-davies) and [Laurent Jacques](https://laurentjacques.gitlab.io/).

We use the [deepinv library](https://deepinv.github.io/deepinv/) for most of the code.


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
\max_{\eta} \min_{f} \sum_{i=1}^N \|y_i - f(y_i)\|^2 + 2\Sigma_{\eta} \text{div} f(y_i) 
$$
where $\eta$ is a Lagrange multiplier, 
$\Sigma_{\eta}$ is the covariance matrix of the noise and $\text{div} f(y)$ is the divergence of the network $f$ at the point $y$.

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
TODO
```