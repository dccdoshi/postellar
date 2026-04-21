from scipy.interpolate import interp1d
import numpy as np

SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))


def gaussian(x, x0=0, sig=1, amp=None):
    # Amplitude term
    if amp is None:
        amp = 1 / np.sqrt(2 * np.pi * sig ** 2)

    return amp * np.exp(-0.5 * ((x - x0) / sig) ** 2)


def add_dv_pad_to_wv_range(dv_pad, wv_range):
    """Extend a wavelength range to make sure it covers the maximum shift `dv_pad` in units of m/s"""
    wv_min, wv_max = wv_range

    # Extend the range to avoid edge effects in the convolution
    wv_shift = calc_shift(dv_pad)
    wv_min /= wv_shift
    wv_max *= wv_shift

    return [wv_min, wv_max]


class SamplingError(ValueError):
    pass


def get_res_from_grid(grid, res_rtol=1e-6):
    # Determine input resolution
    res_grid = grid[:-1] / np.diff(grid)
    res_in = np.median(res_grid)

    # Check if sample at constant resolution
    if (np.abs(res_grid - res_in) / res_in > res_rtol).any():
        raise SamplingError('wv sampling resolution is not constant')

    return res_in


def gauss_convolve(wv, spec, resolution, n_fwhm=7, res_rtol=1e-6, mode='valid', i_plot=True):
    # Get grid resolution
    res_in = get_res_from_grid(wv, res_rtol=res_rtol)

    # Create kernel grid
    # Resolution element for input grid
    res_elem_grid = 1 / res_in
    # Resolution element for kernel
    fwhm = 1 / resolution
    # We want to cover n_fwhm, with the same delta as the input grid
    kernel_grid = np.arange(0, n_fwhm * fwhm / 2 + 0.1 * res_elem_grid, res_elem_grid)
    kernel_grid = np.concatenate([-kernel_grid[-1:0:-1], kernel_grid])

    # Create kernel
    sigma = fwhm / SIGMA_TO_FWHM
    gauss_kernel = gaussian(kernel_grid, sig=sigma)
    gauss_kernel /= gauss_kernel.sum()

    # if i_plot:
    #     # Plot kernel
    #     plt.plot(kernel_grid, gauss_kernel)
    #     plt.axvline(fwhm / 2, linestyle=':')
    #     plt.axvline(-fwhm / 2, linestyle=':', label='fwhm')
    #     plt.axvline(res_elem_grid / 2, linestyle='--')
    #     plt.axvline(-res_elem_grid / 2, linestyle='--', label='grid element')
    #     plt.legend()

    # Convolve
    out_spec = np.convolve(spec, gauss_kernel, mode=mode)

    # Adapt grid to match convolve spectrum
    if mode == 'same':
        wv_out = wv.copy()
    elif mode == 'valid':
        ker_h_len = (gauss_kernel.size - 1) // 2
        wv_out = wv[ker_h_len:-ker_h_len].copy()
    else:
        raise ValueError(f"mode '{mode}' is not implemented.")

    return wv_out, out_spec


def get_wv_constant_res(wv=None, wv_range=None, resolution=None):

    if wv_range is None:
        if wv is None:
            raise ValueError('`wv` must be specified if `wv_range` is not.')
        else:
            wv_range = (np.min(wv), np.max(wv))

    if resolution is None:
        if wv is None:
            raise ValueError('Either `wv` or `resolution` must be specified.')
        else:
            dlog_wv = np.min(np.diff(np.log(wv)))
    else:
        dlog_wv = np.log(1 + 1 / resolution)

    wv_range = np.log(wv_range)
    log_wv = np.arange(wv_range[0], wv_range[-1] + dlog_wv, dlog_wv)

    return np.exp(log_wv)


def resample_constant_res(wv, flux, wv_range=None, resolution=None, kind='cubic', **kwargs):
    flux_spl = interp1d(wv, flux, kind=kind, **kwargs)

    wv_resampled = get_wv_constant_res(wv, wv_range=wv_range, resolution=resolution)
    flux_resampled = flux_spl(wv_resampled)

    return wv_resampled, flux_resampled