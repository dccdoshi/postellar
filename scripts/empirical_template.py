#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

template_file = "../data/Template_s1dv_GL699_sc1d_v_file_AB.fits"

hdu = fits.open(template_file)
data_array = hdu[1].data

wave = data_array['wavelength'].byteswap().newbyteorder()
flux = data_array['flux'].byteswap().newbyteorder()
hdu.close()

mask = ~np.isnan(wave) & ~np.isnan(flux)
wave = wave[mask]
flux = flux[mask]

print(f"Loaded {len(wave)} valid pixels.")
print(f"Wavelength range: {wave[0]:.4f} – {wave[-1]:.4f} nm")

# normalise using the same method as process_one_fits
def normalise_spectrum(wavelength, flux, nbins=150):
    bins = np.linspace(wavelength.min(), wavelength.max(), nbins + 1)
    inds = np.digitize(wavelength, bins)

    w_med = []
    f_med = []
    for i in range(1, nbins + 1):
        bin_mask = (inds == i)
        if np.any(bin_mask):
            w_med.append(np.median(wavelength[bin_mask]))
            f_med.append(np.median(flux[bin_mask]))

    w_med = np.array(w_med)
    f_med = np.array(f_med)

    coeffs = np.polyfit(w_med, f_med, 1)
    continuum = np.polyval(coeffs, wavelength)

    flux_norm = flux / continuum
    flux_norm = flux_norm / np.median(flux_norm)
    return flux_norm

flux_norm = normalise_spectrum(wave, flux)


plt.figure(figsize=(14, 6))
plt.plot(wave, flux, 'k-', lw=1.0)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Raw Flux')
plt.title('Before Normalization)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('empirical_raw.png', dpi=150, bbox_inches='tight')
plt.close()


#normalized spectrum
plt.figure(figsize=(14, 6))
plt.plot(wave, flux_norm, 'k-', lw=1.0)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalised Flux')
plt.title('After Normalization')
plt.grid(True, alpha=0.3)
plt.ylim(0.3, None)   
plt.tight_layout()
plt.savefig('empirical_norm.png', dpi=150, bbox_inches='tight')
plt.close()

#zoomed
plt.figure(figsize=(14, 6))
plt.plot(wave, flux_norm, 'k-', lw=1.0)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalised Flux')
plt.title('After Normalization')
plt.grid(True, alpha=0.3)
plt.ylim(0.35, 1.15)  
plt.xlim(1280, 1321)
plt.tight_layout()
plt.savefig('empirical_norm_zoom.png', dpi=150, bbox_inches='tight')
plt.close()
