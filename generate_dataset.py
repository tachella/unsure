import deepinv as dinv
from helper_fn import get_problem



problem = 'MNIST_denoising1'

# OPTIONS are
# 'MNIST_denoising1' for sigma=0.1
# 'MNIST_denoising2' for sigma=0.2
# 'MNIST_denoising3' for sigma=0.3
# 'MNIST_denoising4' for sigma=0.4
# 'MNIST_denoising5' for sigma=0.5
# 'MRI' for MRI reconstruction with sigma=0.03
#  'CorrelatedNoise_DIV2K' for Div2k denoising with spatially correlated noise with sigma=0.2


_, physics, data_train, data_test, batch_size, device, path = get_problem(problem, generate_dataset=True)

dataset_path = dinv.datasets.generate_dataset(
    data_train,
    physics,
    path,
    test_dataset=data_test,
    device=device,
    train_datapoints=None,
    test_datapoints=None,
    dataset_filename="dinv_dataset",
    batch_size=batch_size,
    num_workers=4,
    supervised=True,
    verbose=True,
    show_progress_bar=True)

data_train = dinv.datasets.HDF5Dataset(path=dataset_path, train=True)

x, y = data_train[0]

y = y.to(device).unsqueeze(0)
x = x.to(device).unsqueeze(0)

xhat = physics.A_dagger(y.to(device))
xhat = xhat.to(x.device)
dinv.utils.plot([x, y, xhat])