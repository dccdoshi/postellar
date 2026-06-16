import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import pandas
from IPython.display import display
import pickle
import h5py
import torch
import sys
sys.path.append('../src')
from transformer import *
from template import Template
from sbart_rv_finder import RV_Retrieval
from convolution import *
from scipy.optimize import minimize_scalar
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.constants import c


#Local copy!!

# Processing part will go in an entirely different script, but I am putting it here for now just to get the data in the right format for POSTELLAR
def process_one_fits(filepath, order=20):
    ''' These are the current pre-processing steps for our Barnard's Star data. '''
    with fits.open(filepath) as hdul:
        primary_header = hdul[0].header
        fluxab_header = hdul[1].header
        
        berv = fluxab_header['BERV']
        snr_key = f'EXTSN{order:03d}'
        snr = fluxab_header[snr_key]
        
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
        'snr': float(snr/np.sqrt(2)),   #we need to divide SNR by root(2) to get per pixel instead of resolution element
        'berv': float(berv),
        'filename': os.path.basename(filepath),
        'sys_velocity': float(sys_velocity),
        'observationID': observation_number}


#path to your folder which contain the data and the files you want to process
data_folder = "../data/Barnard's_Star_Data/selected_observations"
files = sorted(glob.glob(os.path.join(data_folder, "*.fits")))

order = 20
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

with h5py.File(output_file, 'r') as f:
    spectra = torch.tensor(f['spectra'][:])   
    masks = torch.tensor(f['masks'][:])         
    wavelengths = torch.tensor(f['wavelengths'][:])
    berv_km = torch.tensor(f['berv_array'][:])
    snr_values = torch.tensor(f['snr_array'][:])
    sys_values = torch.tensor(f['sys_array'][:])

print(f'The shape of the spectra is: {spectra.shape}')
print(f'The shape of the wavelengths is: {wavelengths.shape}')
print(f'The shape of the BERV array is: {berv_km.shape}')
print(f'The shape of the SNR array is: {snr_values.shape}')
print(f'The shape of the systemic velocity array is: {sys_values.shape}')

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


##### We need to load in the PHOENIX grid to use as the template grid

val_file = f"../data/validation_data/SPIRou{order:02d}_val.df"

with open(val_file, 'rb') as f:
    data = pickle.load(f)

# Get PHOENIX grid (remove padding where wavelength == 1.0)
phoenix_wgrid_padded = data['Wavelength'].iloc[0]
phoenix_wgrid = phoenix_wgrid_padded[phoenix_wgrid_padded != 1.0]

print(f"\n Loaded PHOENIX grid from {val_file}")
print(f"Shape: {phoenix_wgrid.shape}")
print(f"Range: {phoenix_wgrid[0]:.4f} - {phoenix_wgrid[-1]:.4f} nm")

#convert the wavelenght grid to torch tensor
phoenix_wgrid_torch = torch.tensor(phoenix_wgrid, dtype=torch.float64)
print(f"PHOENIX grid shape: {phoenix_wgrid.shape}")

print(wavelengths_2d)

# Prepare inputs
obs_temp = spectra.unsqueeze(0)        # [1, N, L] with L=4088
obs_berv = berv_km * 1000.0              # [N] in m/s
obs_wgrids = wavelengths_2d              # [N, L] – actual grids per observation

# Print wavelength range for each observation
for i in range(len(obs_wgrids)):
    wgrid = obs_wgrids[i].cpu().numpy()
    print(f"Observation {i}: {wgrid[0]:.6f} - {wgrid[-1]:.6f} nm")

# Create template with per-observation grids
template_obj = Template(
    obs_temp=obs_temp,
    obs_berv=obs_berv,                   # BERVs of the observations
    inst_wgrid=None,                     # not used when obs_wgrids provided
    upsampled_wgrid=phoenix_wgrid_torch, # the PHOENIX grid we want to interpolate to
    obs_wgrids=obs_wgrids)               # the actual wavelength grids for each observation

print(f'THIS IS THE OBSERVATION WAVELENGTH GRID', obs_wgrids.shape)

# Here we are making the template
template = template_obj.make_template(func='scipy')

print(f"Template shape: {template.shape}")
print(f"Template has NaNs: {torch.isnan(template).any()}")
print(f'phoenix_wgrid shape: {phoenix_wgrid_torch.shape}')

# Retrieve the BERV-shifted observations (stored in template_obj)
shifted_obs = template_obj.berv_shifted_observations  # shape [1, N, M]
shifted_obs_np = shifted_obs.squeeze(0).cpu().numpy()   # [N, M]

#Plot to make sure the template worked

plt.figure(figsize=(14, 6))
#####################################################
#Plot each shifted observation
#####################################################

for i in range(shifted_obs_np.shape[0]):
    if i==15:
        plt.plot(phoenix_wgrid, shifted_obs_np[i, :], alpha=0.5, lw=1.5, label='Obs 15', c='red')
    else: 
        plt.plot(phoenix_wgrid, shifted_obs_np[i, :], alpha=0.5, lw=0.8)


# Plot template
plt.plot(phoenix_wgrid, template.cpu().numpy(), 'k-', linewidth=1, label='Template')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Flux')
plt.title(f'BERV-Shifted Barnard observations, computing the template - Order {order}')
plt.ylim(0.6, 1.25)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()




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



                            #snr_values[0] is a placeholder        
rv_retrieval = RV_Retrieval(snr_values[0].item(), template, phoenix_wgrid_torch, phoenix_wgrid_torch, len(spectra), "template",  wavelengths_2d)

                                                     #OBS_BERV ARE IN m/s       
                                                     #sig_batch are the flux uncertainties of our observations
planet_rvs, uncs = rv_retrieval.find_dv(data_batch,sig_batch, obs_berv, func='connors')

print("Planet RVs (m/s):", planet_rvs)
print("Uncertainties (m/s):", uncs)


# Number of observations for plotting purposes
n_obs = len(planet_rvs)
indices = np.arange(n_obs)

plt.figure(figsize=(12,5))
plt.errorbar(indices, planet_rvs, yerr=uncs, fmt='o', capsize=3, color='blue', ecolor='gray')
plt.xlabel('Observation index')
plt.ylabel('Planet RV (m/s)')
plt.title(' Radial velocities from template matching')
plt.show()


# SANITY CHECK ON UNCERTAINTIES

def bouchy_uncertainty_from_obs(wavelength, flux, snr, trim_frac=0.01):
    mask = ~np.isnan(flux)

    w_clean = wavelength[mask]
    f_clean = flux[mask]

    n = len(w_clean)
    start = int(trim_frac * n)
    end = int((1 - trim_frac) * n)
    w = w_clean[start:end]
    f = f_clean[start:end]

    A0 = (snr ** 2) * f

    dAdlam = np.gradient(A0, w)

    W = (w * dAdlam) ** 2 / A0
    Q = np.sqrt(np.sum(W)) / np.sqrt(np.sum(A0))
    Ne = np.sum(A0)

    deltaV = c.value / (Q * np.sqrt(Ne))   # m/s
    return deltaV


bouchy_uncs = []
for i in range(len(spectra)):
    w = wavelengths_2d[i].cpu().numpy()
    f = spectra[i].cpu().numpy()
    snr = snr_values[i].item()
    bu = bouchy_uncertainty_from_obs(w, f, snr)
    bouchy_uncs.append(bu)


print("Bouchy uncertainties to compare (m/s):", bouchy_uncs)
