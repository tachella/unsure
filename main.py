import sys
sys.path.append('.')
import deepinv as dinv
import torch
from helper_fn import get_problem, get_losses, CustomTrainer

method = 'divfree_arch'
device = 'cuda:0'
problem = 'DenoisingCorrelated_div2k'
epochs = 100

# set seed
torch.manual_seed(0)

model, physics, data_train, data_test, batch_size, device, path = get_problem(problem, generate_dataset=False)

if hasattr(physics.noise_model, 'sigma'):
    sigma = physics.noise_model.sigma
elif hasattr(physics.noise_model, 'gain'):
    sigma = physics.noise_model.gain
else:
    sigma = 0.

if not isinstance(data_train, list):
    data_train = [data_train]


model, losses = get_losses(method, model, sigma, physics, problem, device)

train_dataloader = [torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True) for data in data_train]
test_dataloader = [torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True) for data in data_test]

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs*.8))


metrics = [dinv.loss.PSNR()]
trainer = CustomTrainer(losses=losses, model=model, ckp_interval=100,
                       physics=physics, verbose_individual_losses=True, metrics=metrics,
                       save_path=path+f'{method}/', online_measurements=False,
                       scheduler=scheduler, optimizer=optimizer, train_dataloader=train_dataloader,
                       device=device, eval_dataloader=test_dataloader, eval_interval=int(epochs/20), epochs=epochs)


trainer.train()
