#!/home/la304/postellar_env_311/bin/python3
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=0-01:00
#SBATCH --account=def-ncowan
#SBATCH --job-name=posterior_checking_30
#SBATCH --output=posterior_checking30%j.out
#SBATCH --error=posterior_checking30%j.err

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import pickle
import h5py
import torch
import sys
from score_models import ScoreModel
sys.path.append('../src')
from transformer import *
from template import Template
from sbart_rv_finder import RV_Retrieval
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.constants import c
from torch.autograd.functional import jacobian
from spectrum_lsf import Score_Likelihood
from mala import MALA

from pathlib import Path

# Processing part will go in an entirely different script, but I am putting it here for now just to get the data in the right format for POSTELLAR
def process_one_fits(filepath, order):
    ''' These are the current pre-processing steps for our Barnard's Star data. '''
    with fits.open(filepath) as hdul:
        primary_header = hdul[0].header
        fluxab_header = hdul[1].header
        berv = fluxab_header['BERV']
        snr_key = f'EXTSN{order:03d}'
        snr = fluxab_header[snr_key]
        print(snr)

        wavelength = hdul['WaveAB'].data[order, :]
        flux = hdul['FluxAB'].data[order, :]
        blaze_correction = hdul['BlazeAB'].data[order, :]
        observation_number = primary_header['OBSID']   
        sys_velocity = fluxab_header['PP_RV']
    # blaze correct
    flux_blaze_corrected = flux / blaze_correction
    
    # remove NaNs
    nan_mask = ~np.isnan(wavelength) & ~np.isnan(flux_blaze_corrected)
    wave_clean = wavelength[nan_mask]
    flux_clean = flux_blaze_corrected[nan_mask]
    
    #Bin the wavelengths and compute the median flux in each. 
    nbins = 150
    bins = np.linspace(wave_clean.min(), wave_clean.max(), nbins + 1)
    inds = np.digitize(wave_clean, bins)
    
    w_med = []; f_med = []
    for i in range(1, nbins + 1):
        bin_mask = (inds == i)
        if np.any(bin_mask):
            w_med.append(np.median(wave_clean[bin_mask]))
            f_med.append(np.median(flux_clean[bin_mask]))
    
    w_med = np.array(w_med)
    f_med = np.array(f_med)
    
    # Fit continuun using a linear polynomial
    coeffs = np.polyfit(w_med, f_med, 1)
    continuum = np.polyval(coeffs, wave_clean)
    
    # Normalize flux
    flux_normalized_clean = flux_clean / continuum
    flux_normalized_clean = flux_normalized_clean / np.median(flux_normalized_clean)
    
    # Reconstruct the array with NaNs
    norm_flux = np.full_like(flux_blaze_corrected, np.nan)
    norm_flux[nan_mask] = flux_normalized_clean
    
    return {
        'spectrum': norm_flux.astype(np.float64),
        'wavelength': wavelength.astype(np.float64),
        'snr': float(snr),
        'berv': float(berv),
        'filename': os.path.basename(filepath),
        'sys_velocity': float(sys_velocity),
        'observationID': observation_number}


#path to your folder which contain the data and the files you want to process
data_folder = f"../data/Barnard_Star_Data/selected_observations"
files = sorted(glob.glob(os.path.join(data_folder, "*.fits")))

order = 28
# Loop through your observation and process each of them. 
# Make a list of observations which each observation is a dictonary
observations = []
for filepath in files:
    print(f'\nProcessing {os.path.basename(filepath)}...')

    obs = process_one_fits(filepath, order=order)
    observations.append(obs)
    print(f'SNR: {obs["snr"]:.1f}')
    print(f'BERV: {obs["berv"]:.2f} km/s')
    print(f'Systemic velocity: {obs["sys_velocity"]:.2f} m/s')
    print(f'ObsID: {obs["observationID"]:.2f}')


    valid = np.sum(~np.isnan(obs['spectrum']))
    print(f'Valid pixels: {valid} out of {len(obs["wavelength"])}')


##################################
# Okay, at this point we have processed our observations. 
# We then want to save them in a format that our code can work with. Save as h5 file

#name of the file to save to
output_file = f'../data/barnards_two_spectra_order_{order}.h5'

# First, prepare the data arrays from your observations list
n_spectra = len(observations)
n_pixels = len(observations[0]['wavelength'])


# Create arrays to store the data
spectra_array = np.zeros((n_spectra, n_pixels))
masks_array = np.zeros((n_spectra, n_pixels), dtype=bool)
wavelengths_array = np.zeros((n_spectra, n_pixels))
berv_array = np.zeros(n_spectra)
snr_array = np.zeros(n_spectra)
sys_array = np.zeros(n_spectra)

# Fill the arrays with your observation data
for i, obs in enumerate(observations):
    spectra_array[i, :] = obs['spectrum']
    masks_array[i, :] = ~np.isnan(obs['spectrum'])  # True for valid pixels
    wavelengths_array[i, :] = obs['wavelength']
    berv_array[i] = obs['berv']
    snr_array[i] = obs['snr']
    sys_array[i] = obs['sys_velocity']

# Now write to the HDF5 file
with h5py.File(output_file, 'w') as f:
    f.create_dataset('spectra', data=spectra_array)
    f.create_dataset('masks', data=masks_array)  #We will see what I need the masks for later but I am saving them just in case
    f.create_dataset('wavelengths', data=wavelengths_array)
    f.create_dataset('berv_array', data=berv_array)  
    f.create_dataset('snr_array', data=snr_array)
    f.create_dataset('sys_array', data=sys_array)
    
print(f"Data saved to {output_file}")

############################################
# WE have now loaded our data in the format we need. 
# Next step is to make a template out of these two observations

with h5py.File(output_file, 'r') as f:
    spectra = torch.tensor(f['spectra'][:])   
    masks = torch.tensor(f['masks'][:])         
    wavelengths_2d = torch.tensor(f['wavelengths'][:])
    berv_km = torch.tensor(f['berv_array'][:])
    snr_values = torch.tensor(f['snr_array'][:])
    sys_values = torch.tensor(f['sys_array'][:])

spectra = spectra.to(DEVICE)
wavelengths_2d = wavelengths_2d.to(DEVICE)
berv_km = berv_km.to(DEVICE)
snr_values = snr_values.to(DEVICE)
sys_values = sys_values.to(DEVICE)

print(f'The shape of the spectra is: {spectra.shape}')
print(f'The shape of the wavelengths is: {wavelengths_2d.shape}')
print(f'The shape of the BERV array is: {berv_km.shape}')
print(f'The shape of the SNR array is: {snr_values.shape}')
print(f'The shape of the systemic velocity array is: {sys_values.shape}')


##### We need to load in the PHOENIX grid to use as the template grid

val_file = f"../data/validation_data/SPIRou{order:02d}_val.df"

with open(val_file, 'rb') as f:
    data = pickle.load(f)

# Get PHOENIX grid (remove padding where wavelength == 1.0)
phoenix_wgrid_padded = data['Wavelength'].iloc[0]

phoenix_wgrid = phoenix_wgrid_padded[phoenix_wgrid_padded != 1.0]
padded_length = len(phoenix_wgrid_padded)
print(f"Padded grid length: {padded_length}")  
training_length = len(data['Final'].iloc[0])
print(f"Training spectrum length: {training_length}")

print(f"\n Loaded PHOENIX grid from {val_file}")
print(f"Shape: {phoenix_wgrid.shape}")
print(f"Range: {phoenix_wgrid[0]:.4f} - {phoenix_wgrid[-1]:.4f} nm")

#convert the wavelenght grid to torch tensor
phoenix_wgrid_torch = torch.tensor(phoenix_wgrid, dtype=torch.float64)
phoenix_wgrid_torch = phoenix_wgrid_torch.to(DEVICE)
print(f"PHOENIX grid shape: {phoenix_wgrid.shape}")
phoenix_wgrid_np = phoenix_wgrid_torch.cpu().numpy()


# Prepare inputs
obs_temp = spectra.unsqueeze(0).to(DEVICE)        # [1, N, L]
obs_berv = (berv_km * 1000.0).to(DEVICE)          # [N] in m/s
obs_wgrids = wavelengths_2d.to(DEVICE)            # [N, L] – actual grids per observation

# Print wavelength range for each observation
for i in range(len(obs_wgrids)):
    wgrid = obs_wgrids[i].cpu().numpy()
    print(f"Observation {i}: {wgrid[0]:.6f} - {wgrid[-1]:.6f} nm")

# Create template with per-observation grids
template_obj = Template(
    obs_temp=obs_temp,
    obs_berv=obs_berv,
    inst_wgrid=None,
    upsampled_wgrid=phoenix_wgrid_torch,
    obs_wgrids=obs_wgrids
)

print(f'THIS IS THE OBSERVATION WAVELENGTH GRID', obs_wgrids.shape)

# Here we are making the template
template = template_obj.make_template(func='scipy')

print(f"Template shape: {template.shape}")
print(f"Template has NaNs: {torch.isnan(template).any()}")
print(f'phoenix_wgrid shape: {phoenix_wgrid_torch.shape}')

# Retrieve the BERV-shifted observations (stored in template_obj)
template_nan = template.clone()
template_nan[template.abs() < 1e-12] = float('nan')
print(f'Template (zeros replaced with NaNs) has NaNs: {torch.isnan(template_nan).any()}')
print(f'Number of NaNs: {torch.isnan(template_nan).sum().item()}')

save_dict = {
    'template': template.cpu(),
    'phoenix_wgrid': phoenix_wgrid_torch.cpu(),
    'obs_wgrids': obs_wgrids.cpu() if obs_wgrids is not None else None,
}
torch.save(save_dict, f'template_data_order_{order}.pt')

shifted_obs = template_obj.berv_shifted_observations  # shape [1, N, M]
shifted_obs_np = shifted_obs.squeeze(0).cpu().numpy()   # [N, M]
print("Template Created")
# Plot to make sure the template worked

plt.figure(figsize=(14, 6))
#####################################################
#Plot each shifted observation
#####################################################
for i in range(shifted_obs_np.shape[0]):
    if i == 15:
        plt.plot(phoenix_wgrid, shifted_obs_np[i, :], alpha=0.5, lw=1.5, label='Obs 15', c='red')
    else:
        plt.plot(phoenix_wgrid, shifted_obs_np[i, :], alpha=0.5, lw=0.8)

# Plot NaN-masked template (zeros become NaN, not plotted)
plt.plot(phoenix_wgrid, template_nan.cpu().numpy(), 'k-', linewidth=1, label='Template (NaNs instead of zeros)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.title(f'BERV-Shifted Barnard observations, computing the template - Order {order}')
plt.ylim(-0.1, 1.25)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'results/plots/observation_template_check_order_{order}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Template plot saved as template_check_order_{order}.png')

print(template_nan.cpu().numpy())
print(len(template_nan.cpu().numpy()))
print(len(phoenix_wgrid))

template_path = f'../data/template_order_{order}.pt'
torch.save(template, template_path)
print(f"Template saved to {template_path}")

######################################################
# Next step is to try and make the RV Retrival work
######################################################
obs_native = spectra   # shape [N, 4088], the native spectra for each observation

#this is the uncertainity grid, currently it is assuming constant uncertainty across all the pixels
#in the observations
sig_native = 1.0 / snr_values.unsqueeze(1) * torch.ones_like(obs_native)  # [N, 4088]

# # Prepare batch dimensions
data_batch = obs_native.unsqueeze(0)   # [1, N, 4088]
sig_batch = sig_native.unsqueeze(0)   # [1, N, 4088]

# Convert only what find_dv expects as numpy
data_batch_np = data_batch.cpu()
sig_batch_np = sig_batch.cpu()
obs_berv_np = obs_berv.cpu().numpy()

# Keep obs_wgrids as a tensor (on CPU is fine; the class will move it if needed)
wavelengths_2d_cpu = wavelengths_2d.cpu()   # tensor, not numpy

rv_retrieval = RV_Retrieval(
    snr_values[0].item(),
    template_nan,                 
    phoenix_wgrid_torch,          
    phoenix_wgrid_torch,         
    len(spectra),
    "template",
    wavelengths_2d_cpu)

#OBS_BERV ARE IN m/s , sig_batch are the flux uncertainties of our observations
planet_rvs, uncs = rv_retrieval.find_dv(data_batch_np, sig_batch_np, obs_berv_np, func='connors')

print("Planet RVs (m/s):", planet_rvs)
print("Uncertainties (m/s):", uncs)

nspec = len(planet_rvs)

# Number of observations for plotting purposes
n_obs = len(planet_rvs)
indices = np.arange(n_obs)

plt.figure(figsize=(12,5))
plt.errorbar(indices, planet_rvs, yerr=uncs, fmt='o', capsize=3, color='blue', ecolor='gray')
plt.xlabel('Observation index')
plt.ylabel('Planet RV (m/s)')
plt.title(' Radial velocities from template matching')

plt.show()

B = 5 #number of posterior samples
nspec = n_obs 

#B copies of the BERVs
bervs_for_sampling = obs_berv.unsqueeze(0).expand(B, nspec)  #[B, N]

#initial planet RVs
AtA_rvs = torch.tensor(planet_rvs, dtype=torch.float64).to(DEVICE)  #[N]

#B copies of initial planet RVs 
planetrv_for_spectrum_sample = AtA_rvs.unsqueeze(0).expand(B, nspec)  #[B, N]

######################################################
 # calculate the information matrix
######################################################

#model and default information
model_name = f"b8nf16ch2_2_2_2_e750_o{order:02d}"
checkpoints_directory = f"../../order_model/{model_name}"
model = ScoreModel(checkpoints_directory=checkpoints_directory, device=DEVICE)
print(f'Model loaded: {model_name}')
gibbs_steps = 1  # number of Gibbs steps. Currently we are just using 1.

#The base flux spectrum
x_ref = torch.load(f'../data/AtA_spectra/AtA_spectrum_{order}.pt', map_location=DEVICE)

phoenix_wgrid_padded_tensor = torch.tensor(phoenix_wgrid_padded, dtype=torch.float64)
non_ones_tensor = torch.where(phoenix_wgrid_padded_tensor != 1.0)[0]
non_ones_start = non_ones_tensor[0].item()
non_ones_end = non_ones_tensor[-1].item() + 1
print(f"non_ones range: {non_ones_start} - {non_ones_end}")


# we now need to loop through each observation and forward model a 
# template spectrum to each RV and BERV value of our observations. We also forward model
# to each varying observational wavelength grid. We then compute the Jacobian of the forward 
# model to the spectrum

list_AtA = []

# The base flux spectrum (reference spectrum)
x = torch.load(f'../data/AtA_spectra/AtA_spectrum_{order}.pt', map_location=DEVICE)

for i in range(nspec):
    print(f'Computing AtA for observation {i}')
    
    # Get velocities for this observation
    planet_chunk = AtA_rvs[i]
    berv_chunk = obs_berv[i]
    sys_vel = sys_values[i]
    
    planetrv_for_A = torch.as_tensor(planet_chunk, device=DEVICE).unsqueeze(0).unsqueeze(0)
    berv_for_A = torch.as_tensor(berv_chunk, device=DEVICE).unsqueeze(0).unsqueeze(0)
    sys_for_A = torch.as_tensor(sys_vel, device=DEVICE).unsqueeze(0).unsqueeze(0)
    
    native_wgrid = wavelengths_2d[i].to(DEVICE)
    
    def f_wrapped(x):
        return forward_model(x, phoenix_wgrid_torch, native_wgrid, 
                             berv_for_A, planetrv_for_A, sys_vel=sys_for_A)
    
    # Compute Jacobian
    A_full = jacobian(f_wrapped, x, create_graph=False)
    
    #extract A
    A = A_full[0, 0, :, 0, 0, :]
    chunk_AtA = torch.matmul(A, A.transpose(-1, -2))
    
    # save each per-observation AtA
    torch.save(chunk_AtA.unsqueeze(0), f'ata_matrix_order_{order}_obs{i}.pt')
    list_AtA.append(chunk_AtA)
    del A_full, A, chunk_AtA
    torch.cuda.empty_cache()

AtA_full = torch.stack(list_AtA, dim=0)  # [N, 4088, 4088]  
print(f"AtA_full shape after cat: {AtA_full.shape}")  # Should be [20, 4088, 4088]

# Also check the first chunk
print(f"list_AtA[0] shape: {list_AtA[0].shape}")      # Should be [4088, 4088]
torch.save(AtA_full, f'ata_matrix_order_{order}.pt')

#Save all data needed for posterior sampling debugging 
debug_data = {
    # Observations
    'spectra': spectra.cpu(),                    # [N, L_obs]
    'wavelengths_2d': wavelengths_2d.cpu(),      # [N, L_obs]
    'obs_berv': obs_berv.cpu(),                  # [N]
    'snr_values': snr_values.cpu(),              # [N]
    'sys_values': sys_values.cpu(),              # [N]
    
    'phoenix_wgrid_padded': phoenix_wgrid_padded,  # [L_padded]
    'phoenix_wgrid': phoenix_wgrid,              # [L_spec]
    'padded_length': padded_length,
    
    'template': template.cpu(),                  # [L_spec]
    
    'AtA_all': AtA_full.cpu(),                  # [N, L_obs, L_obs]
    
    'planet_rvs': torch.tensor(planet_rvs).cpu(),  # [N]
    
    'model_name': model_name,
    'checkpoints_directory': checkpoints_directory,
    
    'order': order,
    'nspec': nspec,}
torch.save(debug_data, f'debug_sampling_data_order_{order}.pt')
print(f"Saved all debug data to debug_sampling_data_order_{order}.pt")

# fill any internal NaNs by interpolating
def fill_internal_nans(obs_flux, native_grid):
    valid_mask = ~torch.isnan(obs_flux)
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return torch.zeros_like(obs_flux), valid_mask
    first_valid = valid_indices[0].item()
    last_valid = valid_indices[-1].item()
    interior_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    interior_mask[first_valid:last_valid+1] = True
    internal_nan_mask = torch.isnan(obs_flux) & interior_mask
    obs_flux_filled = obs_flux.clone()
    if internal_nan_mask.any():
        valid_interior_mask = valid_mask & interior_mask
        valid_wavelengths = native_grid[valid_interior_mask]
        valid_flux = obs_flux[valid_interior_mask]
        if len(valid_wavelengths) > 1:
            interpolated_full = interpolate(
                valid_wavelengths.unsqueeze(0).unsqueeze(0),
                valid_flux.unsqueeze(0).unsqueeze(0),
                native_grid.unsqueeze(0).unsqueeze(0)
            ).squeeze()
            obs_flux_filled[internal_nan_mask] = interpolated_full[internal_nan_mask]
    return obs_flux_filled, torch.isnan(obs_flux)

# first we need to find the most conservative NaN mask
left_edges = []
right_edges = []
for i in range(nspec):
    obs_flux = spectra[i].to(DEVICE)
    native_grid = wavelengths_2d[i].to(DEVICE)
    obs_flux_filled, _ = fill_internal_nans(obs_flux, native_grid)
    valid = ~torch.isnan(obs_flux_filled)
    valid_indices = torch.where(valid)[0]
    if len(valid_indices) > 0:
        left_edges.append(valid_indices[0].item())
        right_edges.append(valid_indices[-1].item())

common_left = max(left_edges)
common_right = min(right_edges)
n_common = common_right - common_left + 1


spectra_stack = []
grids_stack = []
sig_stack = []
AtA_stack = []
berv_stack = []
sys_stack = []
rv_stack = []

# trim the data to that length
for i in range(nspec):
    obs_flux = spectra[i].to(DEVICE)
    native_grid = wavelengths_2d[i].to(DEVICE)
    obs_sig = 1.0 / snr_values.unsqueeze(1) * torch.ones_like(spectra)
    obs_sig = obs_sig[i].to(DEVICE)
    
    obs_flux_filled, _ = fill_internal_nans(obs_flux, native_grid)
    obs_flux_trimmed = obs_flux_filled[common_left:common_right+1]
    grid_trimmed = native_grid[common_left:common_right+1]
    sig_trimmed = obs_sig[common_left:common_right+1]
    
    # Trim AtA 
    AtA_trimmed = AtA_full[i, common_left:common_right+1, common_left:common_right+1]
    
    spectra_stack.append(obs_flux_trimmed.cpu())
    grids_stack.append(grid_trimmed.cpu())
    sig_stack.append(sig_trimmed.cpu())
    AtA_stack.append(AtA_trimmed.cpu())
    berv_stack.append(obs_berv[i].item())
    sys_stack.append(sys_values[i].item())
    rv_stack.append(planet_rvs[i].item())

# Convert to tensors onto device
Y = torch.stack(spectra_stack, dim=0).unsqueeze(0).to(DEVICE)        # [1, N, L_common]
sig_all = torch.stack(sig_stack, dim=0).unsqueeze(0).to(DEVICE)      # [1, N, L_common]
AtA_all = torch.stack(AtA_stack, dim=0).to(DEVICE)                   # [N, L_common, L_common]
grids_all = torch.stack(grids_stack, dim=0).to(DEVICE)               # [N, L_common]
berv_all = torch.tensor(berv_stack, device=DEVICE).unsqueeze(0)      # [1, N]
sys_all = torch.tensor(sys_stack, device=DEVICE).unsqueeze(0)        # [1, N]
V_all = torch.tensor(rv_stack, device=DEVICE).unsqueeze(0)           # [1, N]

print(f"Y shape: {Y.shape}")
print(f"AtA_all shape: {AtA_all.shape}")

LSF = Score_Likelihood(
    Y=Y,
    V=V_all,
    sig_n=sig_all,
    berv=berv_all,
    sys_vel=sys_all,
    spec_wgrid=phoenix_wgrid_torch.to(DEVICE),
    inst_wgrid=grids_all[0],            # fallback (unused if obs_wgrids given)
    non_ones=non_ones_tensor,
    SNR=snr_values,
    beta_min=1e-2,
    beta_max=20,
    AtA=AtA_all,
    obs_wgrids=grids_all)

# sample from the posterior
steps = 10000
print(f"\n Sampling with B={B}, steps={steps}, N={nspec} observations")
posterior_samples = model.sample(
    shape=[B, 1, padded_length],
    steps=steps,
    likelihood_score_fn=LSF
)

# remove padding and save everything
posterior_trimmed = posterior_samples[:, :, non_ones_start:non_ones_end].squeeze(1)
print(f"Posterior trimmed shape: {posterior_trimmed.shape}")

save_data = {
    'posterior_trimmed': posterior_trimmed.cpu(),
    'template': template.cpu(),
    'phoenix_wgrid_np': phoenix_wgrid_torch.cpu().numpy(),
    'non_ones_start': non_ones_start,
    'non_ones_end': non_ones_end,
    'obs_indices': list(range(nspec)),
    'spectra': spectra.cpu(),
    'wavelengths_2d': wavelengths_2d.cpu(),
    'obs_berv': obs_berv.cpu(),
    'planet_rvs': torch.tensor(planet_rvs).cpu(),
    'sys_values': sys_values.cpu(),
}
torch.save(save_data, f'posterior_and_data_order_{order}.pt')

# shift template by -sys_vel to get it to match up with posteriors
sys_vel_shift = sys_values[0].item()  
template_tensor = template.clone().unsqueeze(0).unsqueeze(0).to(DEVICE)
phoenix_wgrid_batched = phoenix_wgrid_torch.unsqueeze(0).unsqueeze(0)

shifted_template_tensor = shift_spectrum(
    template_tensor,
    torch.tensor([[-sys_vel_shift]], device=DEVICE),
    phoenix_wgrid_batched)
shifted_template = shifted_template_tensor.squeeze().cpu().numpy()

plt.figure(figsize=(14, 6))
plt.plot(phoenix_wgrid_np, shifted_template, 'k-', linewidth=1, label='Template (shifted by sys_vel)')
for i in range(min(B, 10)):
    sample = posterior_trimmed[i].cpu().numpy()
    plt.plot(phoenix_wgrid_np, sample, alpha=0.6, linewidth=1, color='blue')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.legend()
plt.ylim(0.35, None)
plt.grid(True, alpha=0.3)
plt.savefig(f'posterior_samples_order_{order}.png', dpi=150, bbox_inches='tight')
plt.close()

# zoomed view
plt.figure(figsize=(14, 6))
plt.plot(phoenix_wgrid_np, shifted_template, 'k-', linewidth=1, label='Template')
for i in range(min(B, 10)):
    sample = posterior_trimmed[i].cpu().numpy()
    plt.plot(phoenix_wgrid_np, sample, alpha=0.6, linewidth=1, color='blue')
plt.xlim(1498, 1506)
plt.ylim(0.35, None)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'posterior_samples_zoom_order_{order}.png', dpi=150, bbox_inches='tight')
plt.close()













# sanity checks with plotting
######
# plt.figure(figsize=(14, 6))

# wavelengths_np = native_grid.cpu().numpy()
# obs_flux_np = obs_flux.cpu().numpy()
# valid_mask_np = valid_mask.cpu().numpy()
# obs_flux_filled_np = obs_flux_filled.cpu().numpy()

# plt.plot(wavelengths_np[valid_mask_np], obs_flux_np[valid_mask_np], 
#          'b.', markersize=1, label='Valid Pixels')

# plt.plot(wavelengths_np, obs_flux_filled_np, 
#          color = 'black', linewidth=0.7, label='Spectrum')

# nan_mask = ~valid_mask_np
# if nan_mask.sum() > 0:
#     plt.scatter(wavelengths_np[nan_mask], obs_flux_filled_np[nan_mask],
#                 color='green', s = 35, marker='x', label='Interpolated pixels')
    
#     # Mark valid region edges
#     if len(valid_indices) > 0:
#         first_valid_idx = valid_indices[0].item()
#         last_valid_idx = valid_indices[-1].item()
#         plt.axvline(wavelengths_np[first_valid_idx], color='gray', linestyle='--', alpha=0.5,
#                     label='Valid region edges')
#         plt.axvline(wavelengths_np[last_valid_idx], color='gray', linestyle='--', alpha=0.5)

# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized Flux')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('obs_filled_check.png', dpi=150, bbox_inches='tight')
# plt.close()

# # ---- ZOOM: 1305 - 1320 nm ----
# zoom_min = 1305.0
# zoom_max = 1320.0
# zoom_mask = (wavelengths_np >= zoom_min) & (wavelengths_np <= zoom_max)

# plt.figure(figsize=(14, 6))
# plt.plot(wavelengths_np[zoom_mask], obs_flux_filled_np[zoom_mask], 
#          'black', linewidth=0.7, label='Spectrum')

# # Valid pixels (blue dots)
# valid_zoom = valid_mask_np[zoom_mask]
# plt.plot(wavelengths_np[zoom_mask][valid_zoom], 
#          obs_flux_np[zoom_mask][valid_zoom], 
#          'b.', markersize=1, label='Valid Pixels')

# # Filled pixels 
# filled_zoom = nan_mask[zoom_mask]
# if filled_zoom.sum() > 0:
#     plt.scatter(wavelengths_np[zoom_mask][filled_zoom], 
#                 obs_flux_filled_np[zoom_mask][filled_zoom],
#                 color='green', s=50, marker='x', label='Interpolated pixels')

# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Normalized Flux')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.savefig('obs_filled_check_zoom_1305_1320.png', dpi=150, bbox_inches='tight')
# plt.close()


# def bouchy_uncertainty_from_obs(wavelength, flux, snr, trim_frac=0.01):
#     mask = ~np.isnan(flux)

#     w_clean = wavelength[mask]
#     f_clean = flux[mask]

#     n = len(w_clean)
#     start = int(trim_frac * n)
#     end = int((1 - trim_frac) * n)
#     w = w_clean[start:end]
#     f = f_clean[start:end]

#     A0 = (snr ** 2) * f

#     dAdlam = np.gradient(A0, w)

#     W = (w * dAdlam) ** 2 / A0
#     Q = np.sqrt(np.sum(W)) / np.sqrt(np.sum(A0))
#     Ne = np.sum(A0)

#     deltaV = c.value / (Q * np.sqrt(Ne))   # m/s
#     return deltaV


# bouchy_uncs = []
# for i in range(len(spectra)):
#     w = wavelengths_2d[i].cpu().numpy()
#     f = spectra[i].cpu().numpy()
#     snr = snr_values[i].item()
#     bu = bouchy_uncertainty_from_obs(w, f, snr)
#     bouchy_uncs.append(bu)


# print("Bouchy uncertainties to compare (m/s):", bouchy_uncs)


