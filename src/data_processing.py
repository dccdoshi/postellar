''' Code to normalize and perform some pre-processing steps before going into the POSTELLAR framework. 
If the normalization of the observations wants to be done in a separate way, those changes can be made in this file 
'''

import numpy as np
import os
import astropy.io.fits as fits

def normalize_flux(wave_clean, flux_clean, nbins=100):
    '''
    Function to normalize the flux of a spectrum by fitting a continuum and dividing it out.
 
    The spectrum is binned and the median of each bin is taken. We then fit a linear line through 
    the median and divide that out.
    Important - both inputs must already have their NaNs removed
    '''
    # Split the range into nbins and compute the median of each bin
    bins = np.linspace(wave_clean.min(), wave_clean.max(), nbins + 1)
    inds = np.digitize(wave_clean, bins)
 
    w_med, f_med = [], []
    for i in range(1, nbins + 1):
        # Select the pixels in that specific bin
        bin_mask = (inds == i)
        # If an interior bin is not empty: (an empty interior bin could exist if there is a large NaN gap in the spectrum) 
        if np.any(bin_mask):
            w_med.append(np.median(wave_clean[bin_mask]))
            f_med.append(np.median(flux_clean[bin_mask]))
 
    w_med = np.array(w_med)
    f_med = np.array(f_med)
 
    # Fit a linear line
    coeffs = np.polyfit(w_med, f_med, 1)
    continuum = np.polyval(coeffs, wave_clean)
 
    # Divide out the linear line and then divide by the median of the spectrum
    flux_normalized_clean = flux_clean / continuum
    flux_normalized_clean = flux_normalized_clean / np.median(flux_normalized_clean)
 
    return flux_normalized_clean
 
 
def process_one_fits(filepath, order):
    '''
    Reads one order from a t.fits file 
    We apply the blaze correction and then normalize the spectrum
    '''
    with fits.open(filepath) as hdul:
        primary_header = hdul[0].header
        fluxab_header = hdul[1].header
 
        # Index by the specified order
        wavelength = hdul['WaveAB'].data[order, :]
        flux = hdul['FluxAB'].data[order, :]
        blaze_correction = hdul['BlazeAB'].data[order, :]
        snr = fluxab_header[f'EXTSN{order:03d}']
 
        # These belong to the entire observation
        berv = fluxab_header['BERV']              # barycentric velocity, km/s
        sys_velocity = fluxab_header['PP_RV']     # systemic velocity, m/s
        observation_number = primary_header['OBSID']
 
    # Divide out the blaze
    flux_blaze_corrected = flux / blaze_correction
 
    # Set the NaNs aside while the continuum is fitted
    nan_mask = ~np.isnan(wavelength) & ~np.isnan(flux_blaze_corrected)
    wave_clean = wavelength[nan_mask]
    flux_clean = flux_blaze_corrected[nan_mask]
 
    flux_normalized_clean = normalize_flux(wave_clean, flux_clean)
 
    # Put the NaNs back into their original position
    norm_flux = np.full_like(flux_blaze_corrected, np.nan)
    norm_flux[nan_mask] = flux_normalized_clean
 
    return {
        'spectrum': norm_flux.astype(np.float64),
        'wavelength': wavelength.astype(np.float64),
        'snr': float(snr),
        'berv': float(berv),
        'sys_velocity': float(sys_velocity),
        'observationID': observation_number,
        'filename': os.path.basename(filepath)}