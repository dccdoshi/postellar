#!/home/la304/postellar_env_311/bin/python3
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=0-01:30
#SBATCH --account=def-ncowan
#SBATCH --job-name=final_run20
#SBATCH --output=final_run20%j.out
#SBATCH --error=final_run20%j.err

"""
Beginning stages of getting the POSTELLAR pipeline to work on real data!
We are currently using SPIRou observations of Barnard's Star. 

In order to run this script you will need: from a scripts/ folder, with:
- Raw FITS files in ../data/Barnard_Star_Data/selected_observations/
- PHOENIX grid in ../data/validation_data/
- Reference spectrum (x_ref) in ../data/AtA_spectra/
- The trained machine learning model in ../../order_model/

The outputs of this pipeline - 
- intermediate .pt files saved in scripts/
- final results in ../results/
These files are all named with the order number, so nothing gets overwritten when running a different order.
"""
import os
import glob
import numpy as np
import pickle
import h5py
import torch
import sys
from score_models import ScoreModel

sys.path.append('../src')
from data_processing import process_one_fits
from transformer import *
from template import Template
from sbart_rv_finder import RV_Retrieval
from torch.autograd.functional import jacobian
from spectrum_lsf import Score_Likelihood
from mala import MALA

########################################################################
# LOAD + PROCESS T.FITS FILES
########################################################################

# Where the selected observations are
data_folder = "../data/Barnard_Star_Data/selected_observations"
# Get all FITS files
files = sorted(glob.glob(os.path.join(data_folder, "*.fits")))

# Choose the order to process
order = 20

# Loop through each file and process it. Returns a dictionary with the processed observational data
observations = []
for filepath in files:
    print(f'\nProcessing {os.path.basename(filepath)}...')

    # Call the data processing function and pass your specific order of interest
    obs = process_one_fits(filepath, order=order)
    observations.append(obs)

    print(f'SNR: {obs["snr"]:.1f}')
    print(f'BERV: {obs["berv"]:.2f} km/s')
    print(f'Systemic velocity: {obs["sys_velocity"]:.2f} m/s')
    print(f'ObsID: {obs["observationID"]}')

    # Count valid pixels in the spectrum
    valid = np.sum(~np.isnan(obs['spectrum']))
    print(f'Valid pixels: {valid} out of {len(obs["wavelength"])}')

########################################################################
# SAVE PROCESSED DATA
########################################################################
output_file = f'../data/barnards_processed_order_{order}.h5'

nspec = len(observations)                  # number of spectra
n_pixels = len(observations[0]['wavelength']) # number of total pixels. Will be the same for each observation (4088)

# Initialize zero array with the appropriate shape to write out data to
spectra_array = np.zeros((nspec, n_pixels))           # each observation's spectra
masks_array = np.zeros((nspec, n_pixels), dtype=bool) # NaN masks for each observation
wavelengths_array = np.zeros((nspec, n_pixels))       # each observation's unique wavelength grid
berv_array = np.zeros(nspec) # BERVs
snr_array = np.zeros(nspec)  # snr for that order
sys_array = np.zeros(nspec)  #systemic velocity

# Fill in the arrays
for i, obs in enumerate(observations):
    spectra_array[i, :] = obs['spectrum']          
    masks_array[i, :] = ~np.isnan(obs['spectrum'])  # True for valid pixels
    wavelengths_array[i, :] = obs['wavelength']
    berv_array[i] = obs['berv']
    snr_array[i] = obs['snr']
    sys_array[i] = obs['sys_velocity']

# Write to the output file. You can later use this for plotting or debugging
with h5py.File(output_file, 'w') as f:
    f.create_dataset('spectra', data=spectra_array)
    f.create_dataset('masks', data=masks_array)
    f.create_dataset('wavelengths', data=wavelengths_array)
    f.create_dataset('berv_array', data=berv_array)
    f.create_dataset('snr_array', data=snr_array)
    f.create_dataset('sys_array', data=sys_array)

print(f"Data saved to {output_file}")

# ONTO THE GPU  
spectra = torch.tensor(spectra_array).to(DEVICE)              # [N, L_obs]
obs_for_temp = spectra.unsqueeze(0).to(DEVICE)                    # [1, N, L_obs]
wavelengths_2d = torch.tensor(wavelengths_array).to(DEVICE)   # [N, L_obs]
berv_km = torch.tensor(berv_array).to(DEVICE)                 # [N] in km/s
obs_berv = (berv_km * 1000.0).to(DEVICE)                      # [N] now in m/s
snr_values = torch.tensor(snr_array).to(DEVICE)               # [N]
sys_values = torch.tensor(sys_array).to(DEVICE)               # [N] in m/s

#########################################################################
# PHOENIX GRID HANDLING
#########################################################################

val_file = f"../data/validation_data/SPIRou{order:02d}_val.df"
with open(val_file, 'rb') as f:
    data = pickle.load(f)

# Get the grid - this grid is padded to be divisible by 32, which was what the model was trained on
phoenix_wgrid_padded = data['Wavelength'].iloc[0]
padded_length = len(phoenix_wgrid_padded)          # Later we require this length for sampling

real_pixels = phoenix_wgrid_padded != 1.0   

phoenix_wgrid = phoenix_wgrid_padded[real_pixels] # Drop the padding, originally required so spectra was divisible by 32
phoenix_wgrid_torch = torch.tensor(phoenix_wgrid, dtype=torch.float64).to(DEVICE)
phoenix_wgrid_np = phoenix_wgrid_torch.cpu().numpy()
print(f"PHOENIX grid for this order: {phoenix_wgrid[0]:.4f} - {phoenix_wgrid[-1]:.4f} nm")

# and where they sit (indices), for slicing the padding back off what the model returns
non_ones_tensor = torch.where(torch.tensor(real_pixels))[0]
non_ones_start = non_ones_tensor[0].item()
non_ones_end = non_ones_tensor[-1].item() + 1

#########################################################################
# TEMPLATE CREATION
#########################################################################
template_obj = Template(obs_temp=obs_for_temp, obs_berv=obs_berv, sys_vel=sys_values, inst_wgrid=None,
    upsampled_wgrid=phoenix_wgrid_torch, obs_wgrids=wavelengths_2d)

template = template_obj.make_template(func='scipy')   #make the template

# Create a copy that we can add an extra trim on
template_nan = template.clone()
print(f"Template shape: {template_nan.shape}")
print(f"Template has NaNs: {torch.isnan(template_nan).any()}")
print(f'Total Number of NaNs in Template: {torch.isnan(template_nan).sum().item()}')

# You can do an extra fractional trim on the Template after it has been created. I have chosen 
# to trim off a total of 1% on the Template to avoid any extra edge effects that didn't get caught.
# This can be removed later if deemed not necessary, this is a safety step

valid_idx = torch.where(~torch.isnan(template_nan))[0]    
v_left, v_right = valid_idx[0].item(), valid_idx[-1].item() # identify first and last valid pixel on Template  

# Trim off an extra 1% on the total Template (0.5% on each side) 
extra_trim = int(0.005 * (v_right - v_left))
v_left, v_right = v_left + extra_trim, v_right - extra_trim

# Pixels marked by trim are now NaNs
template_nan[:v_left] = float('nan')
template_nan[v_right:] = float('nan')
print(f'Template after extra 1% trim: valid range [{v_left}:{v_right}]')

########################################################################
# S-BART: FIT THE TEMPLATE TO EACH OBSERVATION
########################################################################
# Flat 1/SNR per observation. The synthetic pipeline used sqrt(flux), which scales with the flux
# This is likely worth revisiting in the future
sig_native = 1.0 / snr_values.unsqueeze(1) * torch.ones_like(spectra)   # [N, L_obs]

# find_dv runs scipy's minimize_scalar, so things have to be on the CPU. 
data_cpu = spectra.unsqueeze(0).cpu()        # [1, N, L_obs]
sig_cpu = sig_native.unsqueeze(0).cpu()      # [1, N, L_obs]
obs_berv_np = obs_berv.cpu().numpy()         # [N]
sys_values_np = sys_values.cpu().numpy()     # [N]
wavelengths_2d_cpu = wavelengths_2d.cpu()    # [N, L_obs] each observation's own grid

# Note snr_values[0] is stored by the class but not used
rv_retrieval = RV_Retrieval(snr_values[0].item(), template_nan, phoenix_wgrid_torch, None, nspec, "template", wavelengths_2d_cpu)   # instrument_wgrid will be unused

planet_rvs, template_uncs = rv_retrieval.find_dv(data_cpu, sig_cpu, obs_berv_np, func='connors', sys_vel=sys_values_np)
print("Planet RVs (m/s):", planet_rvs)
print("Uncertainties (m/s):", template_uncs)

planet_rvs_torch = torch.tensor(planet_rvs, dtype=torch.float64).to(DEVICE)  # [N]

########################################################################
# MODEL CHECKPOINT
########################################################################
# Order 20's model has e500 whereas all the other models have e750. Be aware of that
model_name = f"b8nf16ch2_2_2_2_e500_o{order:02d}"
checkpoints_directory = f"../../order_model/{model_name}"


########################################################################
# ATA MATRIX COMPUTATION
########################################################################
# Load in the reference spectrum
x_ref = torch.load(f'../data/AtA_spectra/AtA_spectrum_{order:02d}.pt', map_location=DEVICE)

list_AtA = []
for i in range(nspec):
    print(f'Computing AtA for observation {i}')

    planetrv_for_A = planet_rvs_torch[i].view(1, 1)
    berv_for_A = obs_berv[i].view(1, 1)
    sys_for_A = sys_values[i].view(1, 1)
    native_wgrid = wavelengths_2d[i]

    def f_wrapped(spec):
        return forward_model(spec, phoenix_wgrid_torch, native_wgrid, berv_for_A, planetrv_for_A, sys_vel=sys_for_A)

    A_full = jacobian(f_wrapped, x_ref, create_graph=False)
    A = A_full[0, 0, :, 0, 0, :]                      # [L_obs, L_phoenix]
    chunk_AtA = torch.matmul(A, A.transpose(-1, -2))  # [L_obs, L_obs]
    list_AtA.append(chunk_AtA)

    del A_full, A, chunk_AtA
    torch.cuda.empty_cache()

AtA_full = torch.stack(list_AtA, dim=0)               # [N, L_obs, L_obs]
print(f"AtA_full shape: {AtA_full.shape}")
torch.save(AtA_full, f'ata_matrix_order_{order}.pt')

########################################################################
# COMMON WAVELENGTH RANGE ACROSS ALL OBSERVATIONS
########################################################################
# Compute a conservative mask for the data. This mask propagates down the remainder of this pipeline 
# AtA and Y trim below, the posterior's own boundary, and MALA's mask.

# Fill in any internal NaNs
spectra_filled = torch.stack([fill_internal_nans(spectra[i], wavelengths_2d[i]) for i in range(nspec)])
n_filled = int(torch.isnan(spectra).sum() - torch.isnan(spectra_filled).sum())
print(f"Filled {n_filled} interior gap pixels across {nspec} observations")

left_edges, right_edges = [], []
for i in range(nspec):
    valid_indices = torch.where(~torch.isnan(spectra_filled[i]))[0]
    if len(valid_indices) > 0:
        left_edges.append(valid_indices[0].item())
        right_edges.append(valid_indices[-1].item())

# The widest range where every observation has data, then trimmed inward by an additional 3% on each side
common_left = max(left_edges)
common_right = min(right_edges)
common_trim = int(0.03 * (common_right - common_left))
common_left += common_trim
common_right -= common_trim
print(f"Common range after 3% trim: [{common_left}:{common_right}] "
      f"({common_trim} pixels trimmed each side)")

########################################################################
# TRIM EVERYTHING TO THAT RANGE
########################################################################
sl = slice(common_left, common_right + 1)

Y = spectra_filled[:, sl].unsqueeze(0)      # [1, N, L_common]
sig_all = sig_native[:, sl].unsqueeze(0)    # [1, N, L_common]
grids_all = wavelengths_2d[:, sl]           # [N, L_common]
AtA_all = AtA_full[:, sl, sl]               # [N, L_common, L_common]

# Already the right values in the right order, so they only need the batch axis
berv_all = obs_berv.unsqueeze(0)            # [1, N]
sys_all = sys_values.unsqueeze(0)           # [1, N]
V_all = planet_rvs_torch.unsqueeze(0)       # [1, N]

print(f"Y shape: {Y.shape}")
print(f"AtA_all shape: {AtA_all.shape}")

########################################################################
# SAVE EVERYTHING NEEDED TO RESUME FROM HERE
########################################################################
pipeline_state = {
    'AtA_all': AtA_all.cpu(),           # trimmed
    'common_left': common_left,
    'common_right': common_right,
    'snr_values': snr_values.cpu(),
    'phoenix_wgrid_padded': phoenix_wgrid_padded,
    'padded_length': padded_length,
    'order': order,
    'nspec': nspec
    }
torch.save(pipeline_state, f'pipeline_state_order_{order}.pt')
print(f"Saved pipeline state to pipeline_state_order_{order}.pt")


########################################################################
# LOAD IN THE DIFFUSION MODEL
########################################################################
model = ScoreModel(checkpoints_directory=checkpoints_directory, device=DEVICE)
print(f'Model loaded: {model_name}')

########################################################################
# LIKELIHOOD FUNCTION
########################################################################

LSF = Score_Likelihood(Y=Y, V=V_all, sig_n=sig_all, berv=berv_all, sys_vel=sys_all,
    spec_wgrid=phoenix_wgrid_torch, 
    inst_wgrid=grids_all[0],  # fallback, unused since obs_wgrids is given 
    obs_wgrids=grids_all, non_ones=non_ones_tensor, SNR=snr_values,  AtA=AtA_all, beta_min=1e-2, beta_max=20)

########################################################################
# # POSTERIOR SAMPLING
########################################################################
B = 5   # number of posterior samples to generate
steps = 10000
print(f"\nSampling with B={B}, steps={steps}, N={nspec} observations")
posterior_samples = model.sample(shape=[B, 1, padded_length], steps=steps, likelihood_score_fn=LSF)

posterior_trimmed = posterior_samples[:, :, non_ones_start:non_ones_end].squeeze(1) # Remove the padding
print(f"Posterior trimmed shape: {posterior_trimmed.shape}")

# Clean the posterior by replacing the pixels outside the common left/right area with NaNs
# The left/right boundaries are pixel indices from our observations but the posteriors lie on the PHOENIX grid. We need to translate that boundary
c_val = const.c.value
rest_left_edges, rest_right_edges = [], []
for i in range(nspec):
    w_i = wavelengths_2d[i].cpu().numpy()
    shift_i = obs_berv[i].item() - sys_values[i].item()  # explicit raw berv minus sys_vel, per observation
    ratio = np.sqrt((1 - shift_i/c_val) / (1 + shift_i/c_val))
    rest_left_edges.append(w_i[common_left] * ratio)
    rest_right_edges.append(w_i[common_right] * ratio)

# Take the most conservative numbers
rest_left_nm = max(rest_left_edges)
rest_right_nm = min(rest_right_edges)

# Then convert this to an index for the posteriors
post_left = np.searchsorted(phoenix_wgrid_np, rest_left_nm)
post_right = np.searchsorted(phoenix_wgrid_np, rest_right_nm)

# In these observations, there are often more bad pixels on the left than the right. 
# To account for this, I have decided to perform an asymmetric trim and replace more of the left region with NaNs rather than trimming symmetrically
# This step may or may not be correct. I added this in as the common_left and common_right did not seem to be capturing the non-constrained edges of the posteriors
post_trim_left = 0.05
post_trim_right = 0.02

post_width = post_right - post_left   
post_left = post_left + int(post_trim_left * post_width)  #apply the trimming
post_right = post_right - int(post_trim_right * post_width) #apply the trimming
print(f"Posterior boundary (translated from observation coverage): [{post_left}:{post_right}]  "
      f"(trimmed {post_trim_left*100:.0f}% left, {post_trim_right*100:.0f}% right)")

posterior_clean = posterior_trimmed.clone()
posterior_clean[:, :post_left] = float('nan')   # replace with NaNs
posterior_clean[:, post_right:] = float('nan')  # replace with NaNs
n_nan_before = torch.isnan(posterior_trimmed).sum().item()
n_nan_after = torch.isnan(posterior_clean).sum().item()
print(f"Posterior NaN count before cleaning: {n_nan_before}")  #Should be 0
print(f"Posterior NaN count after cleaning: {n_nan_after}")

posterior_trimmed = posterior_clean
print("Posterior cleaned, this version is used for everything downstream")

########################################################################
# S-BART FIT AGAINST THE MEAN POSTERIOR
########################################################################
# Run against the mean of the posterior draws with sbart
posterior_mean_for_sbart = posterior_trimmed.nanmean(dim=0)   # [L_phoenix]

rv_retrieval_meanpost = RV_Retrieval(snr_values[0].item(), posterior_mean_for_sbart, phoenix_wgrid_torch, None, 1, "sample", wavelengths_2d_cpu)
meanpost_rvs, meanpost_uncs = rv_retrieval_meanpost.find_dv(data_cpu, sig_cpu, obs_berv_np, func='connors', sys_vel=sys_values_np)

save_data = {
    'posterior_trimmed': posterior_trimmed.cpu(),
    'template': template.cpu(),
    'phoenix_wgrid_np': phoenix_wgrid_np,
    'non_ones_start': non_ones_start,
    'non_ones_end': non_ones_end,
    'obs_indices': list(range(nspec)),
    'spectra': spectra.cpu(),
    'wavelengths_2d': wavelengths_2d.cpu(),
    'obs_berv': obs_berv.cpu(),
    'sys_values': sys_values.cpu(),
    'planet_rvs': torch.tensor(planet_rvs).cpu(),
    'template_uncs': torch.tensor(template_uncs).cpu(),
    'meanpost_rvs': torch.tensor(meanpost_rvs).cpu(),
    'meanpost_uncs': torch.tensor(meanpost_uncs).cpu(),
}
torch.save(save_data, f'posterior_and_data_order_{order}.pt')

########################################################################
# MALA RV SAMPLING WITH THE POSTERIOR SPECTRA
########################################################################
mala_steps = 1000
burn_in_mala = 500

mask = torch.zeros(n_pixels, dtype=torch.bool, device=DEVICE)
mask[common_left:common_right + 1] = True
mask = mask.unsqueeze(0).unsqueeze(0)
print(f"MALA mask valid pixels: {mask.sum().item()} out of {n_pixels}")

obs_batch = spectra_filled.unsqueeze(0)                                  # [1, N, L_obs]
sig_mala = sig_native.unsqueeze(0)
S = posterior_trimmed                                                    # [B, L_phoenix]

# each of the B chains starts from the S-BART velocities
planetrv_for_spectrum_sample = planet_rvs_torch.unsqueeze(0).expand(B, nspec)   # [B, N]

mala = MALA(obs=obs_batch, sig_n=sig_mala, berv=obs_berv, snr=snr_values,
    inst_wgrid=wavelengths_2d, spec_wgrid=phoenix_wgrid_torch,
    sys_vel=sys_values, mask=mask)

mala_samples, mala_accepted = mala.find_rv(planetrv_for_spectrum_sample, S, steps=mala_steps)
print(f'MALA samples shape: {mala_samples.shape}')

mala_chain = mala_samples[burn_in_mala:].cpu().numpy().reshape(-1, nspec) # Remove the burn-in
mala_rv_means = mala_chain.mean(axis=0)
mala_rv_stds = mala_chain.std(axis=0)
mala_accept_rate = mala_accepted[burn_in_mala:].mean().item()
print(f"MALA acceptance rate: {mala_accept_rate:.3f}") 

print("\n=== Final RV comparison ===")
print("Obs  MALA mean   MALA std   Template   Template unc  MeanPost   MeanPost unc")
for i in range(nspec):
    print(f"{i:2d}  {mala_rv_means[i]:9.2f}  {mala_rv_stds[i]:8.2f}  {planet_rvs[i]:9.2f}  "
          f"{template_uncs[i]:10.2f}  {meanpost_rvs[i]:9.2f}  {meanpost_uncs[i]:12.2f}")

mala_results = {
    'mala_samples': mala_samples.cpu(),
    'mala_accepted': mala_accepted.cpu(),
    'mala_rv_means': mala_rv_means,
    'mala_rv_stds': mala_rv_stds,
    'deltaV_per_obs': mala.deltaV_per_obs.cpu()}
torch.save(mala_results, f'mala_results_order_{order}.pt')
print(f"Saved MALA results to mala_results_order_{order}.pt")

########################################################################
# SAVE RESULTS TO .h5
########################################################################
os.makedirs('results', exist_ok=True)
output_h5 = f'results/barnards_order_{order}.h5'

with h5py.File(output_h5, "w") as f_out:
    group_A = f_out.create_group("Order")
    group_A.attrs["order"] = order
    group_A.create_dataset("phoenix_wgrid", data=phoenix_wgrid_np)
    group_A.create_dataset("wavelengths_2d", data=wavelengths_2d.cpu().numpy())
    group_A.create_dataset("non_ones", data=non_ones_tensor.cpu().numpy())

    # SNR and systemic velocity genuinely vary per observation for real data, so they're saved as full datasets
    group_B = group_A.create_group("Observational Parameters")
    group_B.attrs["nspec"] = nspec
    group_B.create_dataset("snr_values", data=snr_values.cpu().numpy())
    group_B.create_dataset("sys_values", data=sys_values.cpu().numpy())
    group_B.attrs["deltaV"] = mala.deltaV.cpu().item()
    group_B.create_dataset("deltaV_per_obs", data=mala.deltaV_per_obs.cpu().numpy())

    group_C = group_B.create_group("Spectrum")
    group_C.create_dataset("posterior_spectrum_samples", data=posterior_samples.cpu().numpy())
    group_C.create_dataset("posterior_cleaned", data=posterior_trimmed.cpu().numpy())
    group_C.create_dataset("template", data=template.cpu().numpy())
    group_C.create_dataset("mala_samples", data=mala_samples.cpu().numpy())   # [steps+1, B, N]
    group_C.create_dataset("mala_accepted", data=mala_accepted.cpu().numpy())
    group_C.create_dataset("template_rv", data=planet_rvs)
    group_C.create_dataset("template_uncertainty", data=template_uncs)
    group_C.create_dataset("meanpost_rv", data=meanpost_rvs)
    group_C.create_dataset("meanpost_uncertainty", data=meanpost_uncs)

print(f"Saved pipeline results to {output_h5}")