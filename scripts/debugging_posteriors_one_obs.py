#!/usr/bin/env python3
#!/home/la304/postellar_env_311/bin/python3
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-00:45
#SBATCH --account=def-ncowan
#SBATCH --job-name=test_single_obs
#SBATCH --output=test_single_obs%j.out
#SBATCH --error=test_single_obs%j.err

"""
Standalone script testing single‑observation likelihood. This script only works for 
one observation N=1
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from torch import vmap # parallelize a function over its batch dimension
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')
from transformer import *
from torch.func import grad
from score_models import ScoreModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Score_Likelihood_single(Y, V, sig_n, berv, sys_vel, spec_wgrid, inst_wgrid, non_ones,
                            SNR, beta_min, beta_max, AtA, valid_mask=None):
    '''
    This is the score likelihood function class. It's inputs are the set of observations and parameters
    that will be used to define the likelihood score. This is function used to compute the posterior sample
    of the spectrum, keeping the velocities fixed. This uses the Convolved Likelihood for Variance Preserving SDE shown in Noe's Paper

    INPUTS:
    Y: Observations given by a torch tensor of [1, N, L] where N is num of observations and L is length of spectrum (detector)
    V: Vector of velocities given by a torch tensor of [N]
    sig_n: The sqrt(std) of the gaussian noise added to the observations is [1, N, L]

    berv: observations berv
    spec_wgrid: the wavelength grid of the continous spectrum
    inst_wgrid: the wavelength grid of the observation
    non_ones: this is to unpad the wavelength grid
    SNR: snr of observations

    beta_min: the beta_min used to train the model
    beta_max: the beta_max used to train the model

    AAT: the A matrix to trasnform the uncertainty in diffusion model through the transformation of the sample
    REAL DATA UPDATE: takes in sys_velocity to be passed to forward_model
    REAL OBSERVATION UPDATE: Added valid_mask to only use the valid pixels in our
    OUTPUTS:
    score_llk: This returns the function that score_models can use to do posterior sampling
    '''

    def find_sigma_t(t):
        beta_primitive = 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
        mu = torch.exp(-0.5 * beta_primitive)
        std = (1 - mu ** 2).sqrt()

        return std, mu

    def find_Sigma(sigma_t, mu, B, N, L, D):
        '''
        Calcuates the uncertainty matrix used for likelihood calculation
        '''
        # AtA is [1, L, L] for a single observation
        sig_AAt = (sigma_t**2).view(B, 1, 1, 1) * AtA.unsqueeze(0)
        sig_mat = (mu**2).view(B, 1, 1, 1) * torch.diag_embed(sig_n**2).expand(1, N, -1, -1)
        Sigma = sig_AAt + sig_mat
        return Sigma

    def cholesky_fast_single(y, mu, x, sig, mask):
        """
        This calculates the likelihood in an efficient way when we have large matrices to invert
        y: [1, N, L]
        x: [B, N, L]
        sig: [B, N, L, L]

        REAL DATA UPDATE: Uses the valid mask to ensure that only the valid pixels are used the likelihood calc.
        """
        device = x.device
        y = y.to(device)
        x = x.to(device)
        sig = sig.to(device)
        mu = mu.to(device)

        B = x.shape[0]  # batch size

        # uses the valid mask
        mask = mask.to(device)

        y_valid = y[:, :, mask]                 # [1, 1, L_valid]
        x_valid = x[:, :, mask]                 # [B, 1, L_valid]
        sig_valid = sig[:, :, mask][:, :, :, mask]  # [B, 1, L_valid, L_valid]  <-- CORRECT

        y_scaled = y_valid * mu.view(B, 1, 1)

        resid = y_scaled - x_valid
        resid = resid.unsqueeze(-1)

        L_chol = torch.linalg.cholesky(sig_valid)
        z = torch.linalg.solve_triangular(L_chol, resid, upper=False)

        quad = torch.sum(z**2, dim=-2).squeeze(-1)
        logdet = 2 * torch.sum(torch.log(torch.diagonal(L_chol, dim1=-2, dim2=-1)), dim=-1)

        #only count the valid pixels
        const = mask.sum().item() * np.log(2 * np.pi)

        llk = -0.5 * (quad + logdet + const)  # [B, 1]
        return llk

    def likelihood_fn(t, x):
        '''
        This calculates the likelihood using the convolved likelihood approximation of our spectrum sample
        with respect to our observations

        t is in shape [B] where B is num samples
        x is in shape [B,1,D] where B is num of samples and D is length of spectrum (upsampled)
        '''
        B = len(x)
        x_unpad = x[:, :, non_ones[0] : non_ones[-1] + 1]
        D = x_unpad.shape[-1]
        L = Y.shape[-1]
        N = Y.shape[1]  # should be 1

        sigma_t, mu = find_sigma_t(t)
        Sigma = find_Sigma(sigma_t, mu, B, N, L, D)

        transformed_X = forward_model(x_unpad, spec_wgrid, inst_wgrid, berv, V, sys_vel)

        llk = cholesky_fast_single(Y, mu, transformed_X, Sigma, mask=valid_mask) #[B, N]
        print(llk) 
        return llk.sum()

    def score_llk(t, x):
        score = grad(likelihood_fn, argnums=1)(t, x)
        return score

    return score_llk

# load debugging data
debug_file = 'debug_sampling_data_order_20.pt'
data = torch.load(debug_file, map_location='cpu', weights_only=False)

spectra = data['spectra'].to(DEVICE)
wavelengths_2d = data['wavelengths_2d'].to(DEVICE)
obs_berv = data['obs_berv'].to(DEVICE)
snr_values = data['snr_values'].to(DEVICE)
sys_values = data['sys_values'].to(DEVICE)
planet_rvs = data['planet_rvs'].to(DEVICE)

phoenix_wgrid_padded = data['phoenix_wgrid_padded']
phoenix_wgrid = data['phoenix_wgrid']
padded_length = data['padded_length']
template = data['template'].to(DEVICE)

model_name = data['model_name']
checkpoints_directory = data['checkpoints_directory']
order = data['order']
nspec = data['nspec']

phoenix_wgrid_padded_tensor = torch.tensor(phoenix_wgrid_padded, dtype=torch.float64, device=DEVICE)
non_ones_tensor = torch.where(phoenix_wgrid_padded_tensor != 1.0)[0]
non_ones_start = non_ones_tensor[0].item()
non_ones_end = non_ones_tensor[-1].item() + 1
print(f"non_ones range: {non_ones_start} - {non_ones_end}")

# load in the model
model = ScoreModel(checkpoints_directory=checkpoints_directory, device=DEVICE)
print(f"Model loaded: {model_name}")

# get the data for the single observation
test_i = 0
obs_flux = spectra[test_i]
obs_berv_single = obs_berv[test_i].item()
sys_vel_single = sys_values[test_i].item()
init_rv_val = planet_rvs[test_i].item()
init_rv = torch.tensor([[init_rv_val]], dtype=torch.float64, device=DEVICE)
native_grid = wavelengths_2d[test_i]
sig_native = 1.0 / snr_values.unsqueeze(1) * torch.ones_like(spectra)
obs_sig = sig_native[test_i]

# load ata for this observation
ata_file = f'ata_matrix_order_{order}_obs{test_i}.pt'
AtA_single = torch.load(ata_file, map_location=DEVICE)  # [1, 4088, 4088]
print(f"AtA_single shape: {AtA_single.shape}")

#compute a NaN mask for each observation and include a 1% trim
valid_mask = ~torch.isnan(obs_flux)
valid_indices = torch.where(valid_mask)[0]
n_valid = valid_mask.sum().item()
exclude_edges = int(0.01 * n_valid)
if len(valid_indices) > 2 * exclude_edges:
    keep_indices = valid_indices[exclude_edges:-exclude_edges]
    trimmed_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    trimmed_mask[keep_indices] = True
else:
    trimmed_mask = valid_mask

print(f"Original valid pixels: {n_valid}")
print(f"Trimmed valid pixels: {trimmed_mask.sum().item()}")

# #fill in the NaNs
# if len(valid_indices) == 0:
#     obs_flux_filled = torch.full_like(obs_flux, 1.0)
# else:
#     first_valid = valid_indices[0].item()
#     last_valid = valid_indices[-1].item()
#     obs_flux_filled = obs_flux.clone()
#     obs_flux_filled[:first_valid] = 1.0
#     obs_flux_filled[last_valid+1:] = 1.0

#     if first_valid < last_valid:
#         interior_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
#         interior_mask[first_valid:last_valid+1] = True
#         valid_interior_mask = valid_mask & interior_mask
#         valid_wavelengths = native_grid[valid_interior_mask]
#         valid_flux = obs_flux[valid_interior_mask]
#         if len(valid_wavelengths) > 1:
#             interpolated_full = interpolate(
#                 valid_wavelengths.unsqueeze(0).unsqueeze(0),
#                 valid_flux.unsqueeze(0).unsqueeze(0),
#                 native_grid.unsqueeze(0).unsqueeze(0)
#             ).squeeze()
#             nan_interior_mask = torch.isnan(obs_flux) & interior_mask
#             obs_flux_filled[nan_interior_mask] = interpolated_full[nan_interior_mask]

# if torch.isnan(obs_flux_filled).any():
#     obs_flux_filled = torch.nan_to_num(obs_flux_filled, nan=1.0)

# print(f"Filled pixels: {(~valid_mask).sum().item()}")

phoenix_wgrid_tensor = torch.tensor(phoenix_wgrid, dtype=torch.float64, device=DEVICE)

LSF = Score_Likelihood_single(
    Y=obs_flux.unsqueeze(0).unsqueeze(0),          # [1, 1, 4088] – filled
    V=init_rv,
    sig_n=obs_sig.unsqueeze(0).unsqueeze(0),
    berv=obs_berv_single,
    sys_vel=sys_vel_single,
    spec_wgrid=phoenix_wgrid_tensor,
    inst_wgrid=native_grid,
    non_ones=non_ones_tensor,
    SNR=snr_values[test_i].item(),
    beta_min=1e-2,
    beta_max=20,
    AtA=AtA_single,
    valid_mask=trimmed_mask  # [L] 
)
# shift the template to match the observationS
template_tensor = template.clone().unsqueeze(0).unsqueeze(0).to(DEVICE)
phoenix_wgrid_batched = phoenix_wgrid_tensor.unsqueeze(0).unsqueeze(0)
shifted_template_tensor = shift_spectrum(
    template_tensor,
    torch.tensor([[sys_vel_single]], device=DEVICE),
    phoenix_wgrid_batched
)
shifted_template = shifted_template_tensor.squeeze().cpu().numpy()
phoenix_wgrid_np = phoenix_wgrid_tensor.cpu().numpy()

# number of posterior
B = 1
steps = 10000
print(f"\n🔹 Sampling with B={B}, steps={steps}")
posterior_samples = model.sample(
    shape=[B, 1, padded_length],
    steps=steps,
    likelihood_score_fn=LSF
)
print(f"Posterior samples shape: {posterior_samples.shape}")

# trim and get the posteriors
posterior_trimmed = posterior_samples[:, :, non_ones_start:non_ones_end].squeeze(1)

colour = ['blue']
plt.figure(figsize=(14, 6))
for i in range(B):
    sample = posterior_trimmed[i].cpu().numpy()
    plt.plot(phoenix_wgrid_np, sample, alpha=0.3, linewidth=0.8, color=colour[i])
plt.plot(phoenix_wgrid_np, shifted_template, 'k-', linewidth=1.2, label='Template')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.title(f'Posterior Samples vs Template (B={B}, steps={steps})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_single_obs_10000_steps.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved test_single_obs_10000_steps.png")

