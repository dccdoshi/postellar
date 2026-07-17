#!/usr/bin/env python3

'''
This code is for taking a look at our residuals between our true spectrum
of Barnard's star and our template created in POSTELLAR, as well as the 
residuals between our true spectrum and our posterior spectra.

This code does the following:
- load and take a look at previously saved postellar data from a specified order
- compute the most conservative mask on our spectra and apply a trim based off of the
specified fraction. 
- load in and normalize our true spectrum of Barnard's star. This is done using only the 
valid region of our observations ie. after the most conservative NaN mask + trimming is applied.
This is our chosen method as the posteriors are informed over that range. The normalization is 
done in the same way that our data is normalized 
- compute residuals between our true - posterior & true - template. Plots out a histogram of
all the residuals across all current 20 observation. This means that each spectrum needs to be 
correctly shifted to match the observation for residuals are calculated. Residuals only occur in the 
valid region of our observations

'''
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.interpolate import interp1d
from scipy.stats import norm

sys.path.append('../src')
from transformer import shift_spectrum, interpolate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
# ------------------------------------------------------------------ 

# parameters
ORDER = 20
OBS_IDX = 0
ZOOM_RANGE = (1310, 1316) #for the zoomed in plot
TRIM_FRACTION = 0.01 

# ------------------------------------------------------------------ 

#load in posterior data that is previously saved
post_file = f"posterior_and_data_order_{ORDER}.pt"
post_data = torch.load(post_file, map_location=DEVICE, weights_only=False)

#this is for obtaining the SNR values (they were just saved in a separate file atm)
debug_file = f"debug_sampling_data_order_{ORDER}.pt"
debug_data = torch.load(debug_file, map_location='cpu', weights_only=False)

posterior_trimmed = post_data['posterior_trimmed']

# template created from POSTELLAR
template = post_data['template']
phoenix_wgrid_np = post_data['phoenix_wgrid_np']
phoenix_wgrid_torch = torch.tensor(phoenix_wgrid_np, dtype=torch.float64, device=DEVICE)
spectra = post_data['spectra'].to(DEVICE)
wavelengths_2d = post_data['wavelengths_2d'].to(DEVICE)
obs_berv = post_data['obs_berv'].to(DEVICE)
planet_rvs = post_data['planet_rvs'].to(DEVICE)
sys_values = post_data['sys_values'].to(DEVICE)
snr_values = debug_data['snr_values']

# compute our common valid range across the observations
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
print(f"Original common range: {common_left} – {common_right} (length {common_right - common_left + 1})")

#apply the trim fraction
trim_pixels = int(TRIM_FRACTION * (common_right - common_left + 1))
trimmed_left = max(common_left + trim_pixels, common_left)
trimmed_right = min(common_right - trim_pixels, common_right)
print(f"Trimmed common range: {trimmed_left} – {trimmed_right} (length {trimmed_right - trimmed_left + 1})")

#get the common wavelength grid
wmin = wavelengths_2d[0].cpu().numpy()[trimmed_left]
wmax = wavelengths_2d[0].cpu().numpy()[trimmed_right]
print(f"Trimmed wavelength range: {wmin:.4f} – {wmax:.4f} nm")

# load in our true spectrum
template_file = "../data/Template_s1dv_GL699_sc1d_v_file_AB.fits"
print(f"Loading {template_file} ...")
hdu = fits.open(template_file)
data = hdu[1].data
wave = data['wavelength'].byteswap().newbyteorder()
flux = data['flux'].byteswap().newbyteorder()
hdu.close()
mask = ~np.isnan(wave) & ~np.isnan(flux)
wave, flux = wave[mask], flux[mask]
f_interp = interp1d(wave, flux, kind='linear', bounds_error=False, fill_value=0.0)
empirical_on_phoenix = f_interp(phoenix_wgrid_np)

# normalize using linear continuum
# identical to what our pipeline does
mask_range = (phoenix_wgrid_np >= wmin) & (phoenix_wgrid_np <= wmax) & np.isfinite(empirical_on_phoenix)
wave_sub = phoenix_wgrid_np[mask_range]
flux_sub = empirical_on_phoenix[mask_range]
nbins = 150
bins = np.linspace(wmin, wmax, nbins + 1)
inds = np.digitize(wave_sub, bins)
w_med, f_med = [], []
for i in range(1, nbins + 1):
    bin_mask = (inds == i)
    if np.any(bin_mask):
        w_med.append(np.median(wave_sub[bin_mask]))
        f_med.append(np.median(flux_sub[bin_mask]))
coeffs = np.polyfit(np.array(w_med), np.array(f_med), 1)
continuum = np.polyval(coeffs, phoenix_wgrid_np)
empirical_norm = empirical_on_phoenix / continuum
empirical_norm /= np.median(empirical_norm[mask_range])
empirical_tensor = torch.tensor(empirical_norm, dtype=torch.float64, device=DEVICE)

#compute the mean posterior
posterior_mean = posterior_trimmed.mean(dim=0).to(DEVICE)

template_tensor = template.to(DEVICE)

# Plotting everything with reference to an observation
# all the spectra need to line up according to that observation

obs_flux_full = spectra[OBS_IDX]
obs_wave_full = wavelengths_2d[OBS_IDX]
obs_flux_np = obs_flux_full.cpu().numpy()
wavelength_nm = obs_wave_full.cpu().numpy()
obs_wave_tensor = obs_wave_full.to(DEVICE).unsqueeze(0).unsqueeze(0)

berv = obs_berv[OBS_IDX].item()
rv = planet_rvs[OBS_IDX].item()
sys = sys_values[OBS_IDX].item()

#for ease combining the two functions so I don't have to separately call
def shift_and_interp(model, vel):
    shifted = shift_spectrum(model.unsqueeze(0).unsqueeze(0),
        torch.tensor([[vel]], device=DEVICE),
        phoenix_wgrid_torch.unsqueeze(0).unsqueeze(0)).squeeze()
    return interpolate(phoenix_wgrid_torch.unsqueeze(0).unsqueeze(0),
        shifted.unsqueeze(0).unsqueeze(0),
        obs_wave_tensor).squeeze().cpu().numpy()

truth_interp = shift_and_interp(empirical_tensor, berv + rv)
post_interp = shift_and_interp(posterior_mean, berv + rv + sys)
temp_interp = shift_and_interp(template_tensor, berv + rv)

#full spectrum 
plt.figure(figsize=(14, 6))
plt.plot(wavelength_nm, obs_flux_np, 'g-', lw=1.2, label='Observation')
plt.plot(wavelength_nm, truth_interp, 'k-', lw=1.2, label='True Spectrum')
plt.plot(wavelength_nm, post_interp, 'b-', lw=1.2, label='Mean Posterior')
plt.plot(wavelength_nm, temp_interp, 'r-', lw=1.2, label='POSTELLAR template')
plt.axvspan(wmin, wmax, color='gray', alpha=0.15, label='Valid and trimmed region used')
plt.xlabel('Wavelength (nm)') ; plt.ylabel('Normalized Flux')
plt.legend() ; plt.grid(True, alpha=0.3)
plt.ylim(0.35, 1.2)
plt.tight_layout()
plt.savefig(f'spectra_order{ORDER}_obs{OBS_IDX}_full.png', dpi=150, bbox_inches='tight')
plt.close()

# zooming in
plt.figure(figsize=(14, 6))
plt.plot(wavelength_nm, obs_flux_np, 'g-', lw=1.2, label='Observation')
plt.plot(wavelength_nm, truth_interp, 'k-', lw=1.2, label='True Spectrum')
plt.plot(wavelength_nm, post_interp, 'b-', lw=1.2, label='Mean Posterior')
plt.plot(wavelength_nm, temp_interp, 'r-', lw=1.2, label='POSTELLAR template')
plt.xlim(*ZOOM_RANGE)
plt.xlabel('Wavelength (nm)') ; plt.ylabel('Normalized Flux')
plt.legend() ; plt.grid(True, alpha=0.3)
plt.ylim(0.35, 1.2)
plt.tight_layout()
plt.savefig(f'spectra_order{ORDER}_obs{OBS_IDX}_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------ 


# plot out the residuals vs wavelengths so we can see which regions the 
# residuals contribute the most
# this is particularly important as regions were shifting/interpolation artifacts
# are should not contribute to our residuals

#this is of course just for one observation
mask_finite = (np.isfinite(obs_flux_np) & np.isfinite(truth_interp) &
               np.isfinite(post_interp) & np.isfinite(temp_interp))
wave_finite = wavelength_nm[mask_finite]

#compute residuals!
resid_post = truth_interp[mask_finite] - post_interp[mask_finite]
resid_temp = truth_interp[mask_finite] - temp_interp[mask_finite]


#Now plot out an observation so we can ensure that everything is lining up correctly
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                               gridspec_kw={'height_ratios': [2, 1]})
ax1.plot(wavelength_nm, obs_flux_np, 'g-', lw=1.2, label='Observation')
ax1.plot(wavelength_nm, truth_interp, 'k-', lw=1.2, label='True Spectrum')
ax1.plot(wavelength_nm, post_interp, 'b-', lw=1.2, label='Mean Posterior')
ax1.plot(wavelength_nm, temp_interp, 'r-', lw=1.2, label='POSTELLAR template')
ax1.axvspan(wmin, wmax, color='gray', alpha=0.15, label='Trimmed region')
ax1.set_ylabel('Normalized Flux')
ax1.legend() ; ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.35, 1.2)

ax2.axhline(0, color='k', lw=0.8, linestyle='--', alpha=0.7)
ax2.plot(wave_finite, resid_post, 'b-', lw=1.0, alpha=0.8, label='Truth - Posterior')
ax2.plot(wave_finite, resid_temp, 'r-', lw=1.0, alpha=0.8, label='Truth - Template')
ax2.set_xlabel('Wavelength (nm)') ; ax2.set_ylabel('Residual')
ax2.legend(loc='upper right', fontsize='small')
ax2.grid(True, alpha=0.3) ; ax2.set_ylim(-0.05, 0.05)
plt.tight_layout()
plt.savefig(f'residual_vs_wavelength_order{ORDER}_obs{OBS_IDX}.png', dpi=150, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------ 


# now go through all observations, shift everything to match that observation and compute residuls
# we sum all of these up in a histogram
resid_post_all = []
resid_temp_all = []

for i in range(N):
    obs_wave_tensor_i = wavelengths_2d[i].to(DEVICE).unsqueeze(0).unsqueeze(0)
    berv_i = obs_berv[i].item()
    rv_i = planet_rvs[i].item()
    sys_i = sys_values[i].item()

    def shift_and_interp_i(model, vel):
        shifted = shift_spectrum(model.unsqueeze(0).unsqueeze(0),
            torch.tensor([[vel]], device=DEVICE),
            phoenix_wgrid_torch.unsqueeze(0).unsqueeze(0)).squeeze()
        return interpolate(phoenix_wgrid_torch.unsqueeze(0).unsqueeze(0),
            shifted.unsqueeze(0).unsqueeze(0),
            obs_wave_tensor_i).squeeze().cpu().numpy()

    truth_i = shift_and_interp_i(empirical_tensor, berv_i + rv_i)
    post_i = shift_and_interp_i(posterior_mean, berv_i + rv_i + sys_i)
    temp_i = shift_and_interp_i(template_tensor, berv_i + rv_i)

    # here we only care about the residuals in the common region
    truth_trim = truth_i[trimmed_left:trimmed_right+1]
    post_trim = post_i[trimmed_left:trimmed_right+1]
    temp_trim = temp_i[trimmed_left:trimmed_right+1]

    # mask in case any residuals are NaNs or infinity
    mask_finite_i = np.isfinite(truth_trim) & np.isfinite(post_trim) & np.isfinite(temp_trim)
    #append these to the lists getting all of our residuals
    resid_post_all.extend(truth_trim[mask_finite_i] - post_trim[mask_finite_i])
    resid_temp_all.extend(truth_trim[mask_finite_i] - temp_trim[mask_finite_i])

resid_post_all = np.array(resid_post_all)
resid_temp_all = np.array(resid_temp_all)

mean_post = np.mean(resid_post_all)
std_post = np.std(resid_post_all)
mean_temp = np.mean(resid_temp_all)
std_temp = np.std(resid_temp_all)

print(f"Truth - Posterior: mean = {mean_post:.4f}, std = {std_post:.4f}")
print(f"Truth - Template: mean = {mean_temp:.4f}, std = {std_temp:.4f}")
print(f"Expected noise std (1/mean SNR): {1.0 / np.mean(snr_values.cpu().numpy()):.4f}")

#plot in a histogram
bin_width = 0.002
bins = np.arange(-0.05, 0.05 + bin_width, bin_width)
weights_post = np.ones_like(resid_post_all) / len(resid_post_all)
weights_temp = np.ones_like(resid_temp_all) / len(resid_temp_all)

x = np.linspace(-0.05, 0.05, 300)
#adding a Gaussian fit on top of each
gauss_post = norm.pdf(x, mean_post, std_post) * bin_width
gauss_temp = norm.pdf(x, mean_temp, std_temp) * bin_width

plt.figure(figsize=(10,6))
plt.hist(resid_post_all, bins=bins, alpha=0.6, weights=weights_post,
         color='blue', label='Truth - Posterior')
plt.hist(resid_temp_all, bins=bins, alpha=0.6, weights=weights_temp,
         color='red', label='Truth - Template')
plt.plot(x, gauss_post, 'b--', lw=2, label=f'Posterior fit (sigma={std_post:.4f})')
plt.plot(x, gauss_temp, 'r--', lw=2, label=f'Template fit (sigma={std_temp:.4f})')
plt.xlabel('Residuals (truth - model)') ; plt.ylabel('Fraction per bin')
plt.title(f'Order {ORDER}')
plt.legend() ; plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 0.05)
plt.tight_layout()
plt.savefig('truth_vs_models_histograms_weights_trimmed.png', dpi=150, bbox_inches='tight')
plt.close()


