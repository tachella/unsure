import deepinv as dinv
import torch
from helper_fn import get_problem, get_losses

# MAKE SURE to generate the dataset before running this script, using generate_dataset.py

# CHOOSE TRAINING METHOD
# 'unsure' for isotropic UNSURE
# 'unsure_score' for UNSURE via score
# 'unsure_corr' for UNSURE with correlated noise
# 'unsure_poisson_gaussian' for UNSURE with Poisson-Gaussian noise
# 'splitting' for cross-validation methods (Noise2Void, Noise2Inverse)
# 'sup' for supervised training
method = 'sure'

# CHOOSE DEVICE
device = 'cuda:0'

# CHOOSE INVERSE PROBLEM
# 'MNIST_denoising1' for sigma=0.1
# 'MNIST_denoising2' for sigma=0.2
# 'MNIST_denoising3' for sigma=0.3
# 'MNIST_denoising4' for sigma=0.4
# 'MNIST_denoising5' for sigma=0.5
# 'MRI' for single-coil MRI reconstruction with sigma=0.03
#  'CorrelatedNoise_DIV2K' for DIV2K denoising with spatially correlated noise with sigma=0.2
problem = 'MNIST_denoising1'


# number of training epochs
epochs = 100

# set seed
torch.manual_seed(0)

model, physics, data_train, data_test, batch_size, device, path = get_problem(problem, generate_dataset=False)

if not isinstance(data_train, list):
    data_train = [data_train]

model, losses = get_losses(method, model, physics, problem, device)

train_dataloader = [torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True) for data in data_train]
test_dataloader = [torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True) for data in data_test]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))


metrics = [dinv.loss.PSNR()]
trainer = dinv.Trainer(losses=losses, model=model, ckp_interval=100,
                       physics=physics, verbose_individual_losses=True, metrics=metrics,
                       save_path=path+f'{method}/', online_measurements=False,
                       scheduler=scheduler, optimizer=optimizer, train_dataloader=train_dataloader,
                       device=device, eval_dataloader=test_dataloader, eval_interval=int(epochs/20), epochs=epochs)


trainer.train()
