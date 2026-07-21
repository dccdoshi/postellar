"""
Attempt at getting the MALA RV retrieval to work with real data
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../src')
from transformer import shift_spectrum, interpolate
from mala import MALA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

#choosing the order
ORDER = 20

# previously set parameters from original pipeline
STEPS = 1000
BURN = 500
TRIM_FRACTION = 0.01

# load in data, again previously computed posteriors
post_data = torch.load(f"posterior_and_data_order_{ORDER}.pt", map_location=DEVICE, weights_only=False)

#for snr values
debug_data = torch.load(f"debug_sampling_data_order_{ORDER}.pt", map_location='cpu', weights_only=False)

posterior_trimmed = post_data['posterior_trimmed']
phoenix_wgrid_np = post_data['phoenix_wgrid_np']
phoenix_wgrid_torch = torch.tensor(phoenix_wgrid_np, dtype=torch.float64, device=DEVICE)
spectra = post_data['spectra'].to(DEVICE)
wavelengths_2d = post_data['wavelengths_2d'].to(DEVICE)   # shape [N_obs, N_pix]
obs_berv = post_data['obs_berv'].to(DEVICE)      # m/s
planet_rvs = post_data['planet_rvs'].to(DEVICE)  # m/s
sys_values = post_data['sys_values'].to(DEVICE)  # m/s
snr_values = debug_data['snr_values']


# Compute common range + 1% trim (same as residual analysis)
N = len(spectra)
left_edges, right_edges = [], []
for i in range(N):
    obs_flux = spectra[i]
    valid_mask = ~torch.isnan(obs_flux)
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        continue
    left_edges.append(valid_indices[0].item())
    right_edges.append(valid_indices[-1].item())

common_left = max(left_edges)
common_right = min(right_edges)
trim_pixels = int(TRIM_FRACTION * (common_right - common_left + 1))
trimmed_left = max(common_left + trim_pixels, common_left)
trimmed_right = min(common_right - trim_pixels, common_right)
print(f"Trimmed range: {trimmed_left} – {trimmed_right}")

wmin = wavelengths_2d[0].cpu().numpy()[trimmed_left]
wmax = wavelengths_2d[0].cpu().numpy()[trimmed_right]

# At the moment just getting it to work using the mean posterior
model = posterior_trimmed.mean(dim=0).to(DEVICE)
S = model.unsqueeze(0).unsqueeze(0)


# perform the MALA sampling
all_samples = []
for i in range(N):
    sys_values_i = sys_values[i].item()
    snr_i = snr_values[i].item()
    obs_i = spectra[i].unsqueeze(0).unsqueeze(0)
    sig_i = (1.0 / snr_values[i]).to(DEVICE).unsqueeze(0).unsqueeze(0).expand_as(obs_i)
    # get each observations wavelength grid
    inst_wgrid_i = wavelengths_2d[i].clone().detach().to(DEVICE)
    berv_i = obs_berv[i].unsqueeze(0)   # m/s 
    x_init = planet_rvs[i].clone().detach().view(1,1)   # m/s

    # get the valid pixel mask
    finite_mask = torch.isfinite(obs_i)

    #and then the trimming mask
    range_mask = torch.zeros(obs_i.shape[-1], dtype=torch.bool, device=DEVICE)
    range_mask[trimmed_left:trimmed_right+1] = True
    range_mask = range_mask.unsqueeze(0).unsqueeze(0)

    #combine into one mask
    mask = finite_mask & range_mask

    # Pass berv and sys_vel separately
    mala = MALA(obs_i, sig_i, berv_i, snr_i, inst_wgrid_i, phoenix_wgrid_torch,
                sys_vel=sys_values_i, mask=mask)
    samples, _ = mala.find_rv(x_init, S, steps=STEPS)
    all_samples.append(samples[:, 0, 0].cpu().numpy())

#throw away the first 500 and then compute means and std deviations
rv_means = [np.mean(s[BURN:]) for s in all_samples]
rv_stds  = [np.std(s[BURN:]) for s in all_samples]

plt.figure(figsize=(10,6))
plt.errorbar(np.arange(N), rv_means, yerr=rv_stds, fmt='o', capsize=3, ecolor='gray', color='blue')
plt.xlabel('Observation index')
plt.ylabel('RV (m/s)')
plt.title(f'Radial velocities from MALA')
plt.tight_layout()
plt.savefig(f'rv_curve_order{ORDER}.png', dpi=150)
plt.close()

for i in range(len(rv_means)):
    print(f"{i}: {rv_means[i]:.2f} +/- {rv_stds[i]:.2f}")
