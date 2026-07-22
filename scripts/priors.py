"""
Code to generate priors with the same number of steps as the posteriors are generated with
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../src')
from transformer import shift_spectrum, interpolate
from score_models import ScoreModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

# parameters
order = 20
B = 5                     # number of samples to generate
steps = 10000             # same as the number of posterior sampling steps

#load in the previously saved data with the posteriors
post_data = torch.load(f"posterior_and_data_order_{order}.pt", map_location='cpu', weights_only=False)
debug_data = torch.load(f"debug_sampling_data_order_{order}.pt", map_location='cpu', weights_only=False)

# extract necessary info
phoenix_wgrid_np = post_data['phoenix_wgrid_np']
phoenix_wgrid_torch = torch.tensor(phoenix_wgrid_np, dtype=torch.float64).to(DEVICE)
posterior_trimmed = post_data['posterior_trimmed']      # [B, L_spec]

#the model
model_name = debug_data['model_name']
checkpoints_directory = debug_data['checkpoints_directory']

# wavelength grid information
phoenix_wgrid_padded = debug_data['phoenix_wgrid_padded']
non_ones_tensor = torch.where(torch.tensor(phoenix_wgrid_padded) != 1.0)[0]
non_ones_start = non_ones_tensor[0].item()
non_ones_end = non_ones_tensor[-1].item() + 1
padded_length = len(phoenix_wgrid_padded)

# load in the model
model = ScoreModel(checkpoints_directory=checkpoints_directory, device=DEVICE)

#generate the priors
prior_samples = model.sample(shape=[B, 1, padded_length], steps=steps, likelihood_score_fn=None) 

prior_trimmed = prior_samples[:, :, non_ones_start:non_ones_end].squeeze(1)

# compute prior mean and the standard deviation
prior_mean = prior_trimmed.mean(dim=0).cpu().numpy()
prior_std = prior_trimmed.std(dim=0).cpu().numpy()

posterior_mean = posterior_trimmed.mean(dim=0).cpu().numpy() #mean posterior

# plot of mean prior with 1 sigma envelope
plt.figure(figsize=(14, 6))
plt.plot(phoenix_wgrid_np, prior_mean, c = 'dodgerblue', lw=2, label='Prior mean')
plt.fill_between(phoenix_wgrid_np, prior_mean - prior_std, prior_mean + prior_std,
                 color='dodgerblue', alpha=0.3)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.title(f'Priors for order {order}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.35, 1.2)
plt.savefig('prior_mean_envelope.png', dpi=150)
plt.close()


#plot the mean prior with the mean posterior for comparison
plt.figure(figsize=(14, 6))
plt.plot(phoenix_wgrid_np, prior_mean, color='dodgerblue', lw=1, label='Prior mean')
plt.plot(phoenix_wgrid_np,posterior_mean,color='black',lw=1,label='Posterior mean')
plt.xlabel('Wavelength (nm)') ; plt.ylabel('Normalized Flux')
plt.title(f'Prior vs Posterior mean for order {order}')
plt.legend() ; plt.grid(True, alpha=0.3)
plt.ylim(0.35, 1.2)
plt.savefig('prior_vs_posterior_mean.png', dpi=150)
plt.close()

