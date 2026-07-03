#!/usr/bin/env python3
#!/home/la304/postellar_env_311/bin/python3
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-01:00
#SBATCH --account=def-ncowan
#SBATCH --job-name=test_multi_obs
#SBATCH --output=test_multi_obs%j.out
#SBATCH --error=test_multi_obs%j.err

"""
Standalone script for any number of observations (N >= 1).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')

from transformer import *
from torch.func import grad
from score_models import ScoreModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Score_Likelihood_loop(Y, V, sig_n, berv, sys_vel, spec_wgrid, inst_wgrid, non_ones,
                          SNR, beta_min, beta_max, AtA, obs_wgrids=None, valid_mask=None):

    def find_sigma_t(t):
        beta_primitive = 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
        mu = torch.exp(-0.5 * beta_primitive)
        std = (1 - mu ** 2).sqrt()
        return std, mu

    def find_Sigma(sigma_t, mu, B, N, L, D):
        sig_AAt = (sigma_t**2).view(B, 1, 1, 1) * AtA.unsqueeze(0)  # [B, N, L, L]
        sig_mat = (mu**2).view(B, 1, 1, 1) * torch.diag_embed(sig_n**2).expand(1, N, -1, -1)
        Sigma = sig_AAt + sig_mat
        return Sigma

    def cholesky_fast_loop(y, mu, x, sig, mask):
        device = x.device
        y = y.to(device)
        x = x.to(device)
        sig = sig.to(device)
        mu = mu.to(device)

        B, N, L_full = x.shape
        y_scaled = y * mu.view(B, 1, 1)   # [B, N, L]

        llk_total = 0.0

        for n in range(N):
            mask_obs = mask[n]                     # [L]
            if mask_obs.sum() == 0:
                continue

            y_valid = y_scaled[:, n, mask_obs]         # [B, L_valid]
            x_valid = x[:, n, mask_obs]                # [B, L_valid]
            sig_valid = sig[:, n, mask_obs][:, :, mask_obs]  # [B, L_valid, L_valid]

            resid = y_valid - x_valid
            resid = resid.unsqueeze(-1)

            L_chol = torch.linalg.cholesky(sig_valid)
            z = torch.linalg.solve_triangular(L_chol, resid, upper=False)

            quad = torch.sum(z**2, dim=-2).squeeze(-1)   # [B]
            logdet = 2 * torch.sum(torch.log(torch.diagonal(L_chol, dim1=-2, dim2=-1)), dim=-1)  # [B]
            const = len(y_valid) * np.log(2 * np.pi)

            llk_total += -0.5 * (quad + logdet + const)  # [B]

        return llk_total

    def likelihood_fn(t, x):
        B = len(x)
        x_unpad = x[:, :, non_ones[0] : non_ones[-1] + 1]
        D = x_unpad.shape[-1]
        L = Y.shape[-1]
        N = Y.shape[1]

        sigma_t, mu = find_sigma_t(t)
        Sigma = find_Sigma(sigma_t, mu, B, N, L, D)

        transformed_X = forward_model(x_unpad, spec_wgrid, inst_wgrid, berv, V, sys_vel,
                                      obs_wgrids=obs_wgrids)

        llk = cholesky_fast_loop(Y, mu, transformed_X, Sigma, mask=valid_mask)
        print(f'likelihood shape: {llk.shape}')
        print('llk is', llk)
        print('ll sum is', llk.sum())
        return llk.sum()

    def score_llk(t, x):
        score = grad(likelihood_fn, argnums=1)(t, x)
        return score

    return score_llk


print(f"Using device: {DEVICE}")

#load the data 
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

#define the non_ones
phoenix_wgrid_padded_tensor = torch.tensor(phoenix_wgrid_padded, dtype=torch.float64, device=DEVICE)
non_ones_tensor = torch.where(phoenix_wgrid_padded_tensor != 1.0)[0]
non_ones_start = non_ones_tensor[0].item()
non_ones_end = non_ones_tensor[-1].item() + 1
print(f"non_ones range: {non_ones_start} - {non_ones_end}")

model = ScoreModel(checkpoints_directory=checkpoints_directory, device=DEVICE)
print(f"Model loaded: {model_name}")

#choose the observations
obs_indices = [0, 1]   # first two observations
N = len(obs_indices)
print(f"Using {N} observations: {obs_indices}")

#stack the data
spectra_stack = []
masks_stack = []
sig_stack = []
AtA_stack = []
berv_stack = []
sys_stack = []
rv_stack = []
snr_stack = []
grids_stack = []

#loop through each observation 
for i in obs_indices:
    obs_flux = spectra[i]
    obs_berv_single = obs_berv[i].item()
    sys_vel_single = sys_values[i].item()
    init_rv_val = planet_rvs[i].item()
    native_grid = wavelengths_2d[i]

    sig_native = 1.0 / snr_values.unsqueeze(1) * torch.ones_like(spectra)
    obs_sig = sig_native[i]

    #get each observations mask and add the 1% trim
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

    # # ---- Fill NaNs (edges → 1, interior interpolated) ----
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

    # Load AtA for this observation
    ata_file = f'ata_matrix_order_{order}_obs{i}.pt'
    AtA_single = torch.load(ata_file, map_location=DEVICE)  # [1, 4088, 4088]

    spectra_stack.append(obs_flux)
    masks_stack.append(trimmed_mask)
    sig_stack.append(obs_sig)
    AtA_stack.append(AtA_single)
    berv_stack.append(obs_berv_single)
    sys_stack.append(sys_vel_single)
    rv_stack.append(init_rv_val)
    snr_stack.append(snr_values[i].item())
    grids_stack.append(native_grid)

#stack into the tensors
Y = torch.stack(spectra_stack, dim=0).unsqueeze(0)          # [1, N, L]
mask_all = torch.stack(masks_stack, dim=0)                  # [N, L]
sig_all = torch.stack(sig_stack, dim=0).unsqueeze(0)        # [1, N, L]
AtA_all = torch.cat(AtA_stack, dim=0)                       # [N, L, L]
berv_all = torch.tensor(berv_stack, device=DEVICE).unsqueeze(0)   # [1, N]
sys_all = torch.tensor(sys_stack, device=DEVICE).unsqueeze(0)     # [1, N]
V_all = torch.tensor(rv_stack, device=DEVICE).unsqueeze(0)        # [1, N]
SNR_all = torch.tensor(snr_stack, device=DEVICE)                  # [N]
grids_all = torch.stack(grids_stack, dim=0)                      # [N, L]

# Check if AtA (and thus Sigma) is diagonal
AtA_0 = AtA_single[0]  # shape [L, L]


phoenix_wgrid_tensor = torch.tensor(phoenix_wgrid, dtype=torch.float64, device=DEVICE)

#likelihood calculation
LSF = Score_Likelihood_loop(
    Y=Y,
    V=V_all,
    sig_n=sig_all,
    berv=berv_all,
    sys_vel=sys_all,
    spec_wgrid=phoenix_wgrid_tensor,
    inst_wgrid=grids_all[0],          # fallback
    non_ones=non_ones_tensor,
    SNR=SNR_all,
    beta_min=1e-2,
    beta_max=20,
    AtA=AtA_all,
    obs_wgrids=grids_all,              # [N, L] per-observation grids
    valid_mask=mask_all                # [N, L]
)

template_tensor = template.clone().unsqueeze(0).unsqueeze(0).to(DEVICE)
phoenix_wgrid_batched = phoenix_wgrid_tensor.unsqueeze(0).unsqueeze(0)
shifted_template_tensor = shift_spectrum(
    template_tensor,
    torch.tensor([[sys_stack[0]]], device=DEVICE),   # use sys 
    phoenix_wgrid_batched
)
shifted_template = shifted_template_tensor.squeeze().cpu().numpy()
phoenix_wgrid_np = phoenix_wgrid_tensor.cpu().numpy()

# posterior sampling
B = 3
steps = 10000
print(f"\n🔹 Sampling with B={B}, steps={steps}, N={N} observations")
posterior_samples = model.sample(
    shape=[B, 1, padded_length],
    steps=steps,
    likelihood_score_fn=LSF
)
print(f"Posterior samples shape: {posterior_samples.shape}")
colours = ['red', 'green', 'yellow']
posterior_trimmed = posterior_samples[:, :, non_ones_start:non_ones_end].squeeze(1)

plt.figure(figsize=(14, 6))
for i in range(B):
    sample = posterior_trimmed[i].cpu().numpy()
    plt.plot(phoenix_wgrid_np, sample, alpha=0.6, linewidth=1, color='blue')
plt.plot(phoenix_wgrid_np, shifted_template, 'k-', linewidth=1, label='Template')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.title(f'Posterior Samples vs Template (B={B}, steps={steps}, N={N} obs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'test_multi_obs_N{N}_steps{steps}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved test_multi_obs_N{N}_steps{steps}.png")
