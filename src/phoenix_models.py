import os
import shutil
import urllib.request
from contextlib import closing
from itertools import product
from pathlib import Path
from typing import Dict, Union
import logging
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

from convolution import resample_constant_res, gauss_convolve, get_wv_constant_res

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()

# try:
#     base_dir = os.environ['SCRATCH']
# except KeyError:
base_dir = "../data/"

DEFAULT_MODEL_DIR = Path(base_dir) / Path("Models/PHOENIX_HiRes/")
print(DEFAULT_MODEL_DIR)
MODEL_SUB_DIR = Path('PHOENIX-ACES-AGSS-COND-2011/')
URL_ROOT = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/"
WAVE_FILENAME = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

# Models available
PHOENIX_GRID = dict()
PHOENIX_GRID['metal'] = np.array([-4, -3, -2, -1.5, -1, -0.5, 0.0, 0.5, 1])
PHOENIX_GRID['alpha'] = np.arange(-0.2, 1.21, 0.2)
PHOENIX_GRID['teff'] = np.append(np.arange(2300, 7000, 100),
                                 np.arange(7000, 12001, 200))
PHOENIX_GRID['logg'] = np.arange(0, 6, 0.5)

MODEL_SUB_DIR = Path('PHOENIX-ACES-AGSS-COND-2011/')
FILE_FRAME = 'lte{:05g}-{:1.2f}{:+1.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
FILE_FRAME_ALPHA = 'lte{:05g}-{:1.2f}{:+1.1f}.Alpha={:+1.2f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
METAL_DIR_FRAME = 'Z{:+1.1f}'
METAL_DIR_FRAME_ALPHA = 'Z{:+1.1f}.Alpha={:+1.2f}'

PHOENIX_RESOLUTION = {(0.05, 0.3):5000,
                      (0.3, 2.5): 5e5,
                      (2.5, 5.5): 2e5}


def download_ftp_file(url, file_save_name):
    with closing(urllib.request.urlopen(url)) as r:
        with open(file_save_name, 'wb') as f:
            shutil.copyfileobj(r, f)

    return

def get_phoenix_grid_in_range(teff=6000, logg=4.5, metal=0.0, alpha=0.0, extrapolate=False):

    # Put input kwargs in a dictionnary with same keys as PHOENIX_GRID
    input_dict = dict(teff=teff, logg=logg, metal=metal, alpha=alpha)

    # Change all inputs to list of 2 elements (min and max)
    # and check if they fall within PHOENIX_GRID
    grid_in_range = dict()
    for key, value in input_dict.items():
        max_val = np.max(value)
        max_grid = PHOENIX_GRID[key].max()
        if max_val > max_grid:
            if not extrapolate:
                msg = f'Max range for {key} ({max_val}) is higher then phoenix grid ({max_grid})'
                raise ValueError(msg)
            max_val = np.clip(max_val, None, max_grid)

        min_val = np.min(value)
        min_grid = PHOENIX_GRID[key].min()
        if min_val < min_grid:
            if not extrapolate:
                msg = f'Min range for {key} ({min_val}) is higher then phoenix grid ({min_grid})'
                raise ValueError(msg)
            min_val = np.clip(min_val, min_grid, None)

        # Find index in PHOENIX_GRID
        max_idx = np.searchsorted(PHOENIX_GRID[key], max_val, side='left')
        min_idx = np.searchsorted(PHOENIX_GRID[key], min_val, side='right') - 1

        grid_in_range[key] = PHOENIX_GRID[key][min_idx:max_idx + 1]

    return grid_in_range


def get_phoenix_filepath(teff=6000, logg=4.5, metal=0.0, alpha=0.0):

    # Little hack to force a - sign before metallicity when == 0.
    if metal == 0.0:
        metal = -1.0e-8

    if alpha == 0.0:
        filename = Path(FILE_FRAME.format(teff, logg, metal))
        file_dir = METAL_DIR_FRAME.format(metal)

    # Alpha element abundances [alpha/M]!=0 are available for -3.0≤[Fe/H]≤0.0. only.
    elif (-3.0 <= metal) and (metal <= 0.0):
        filename = Path(FILE_FRAME_ALPHA.format(teff, logg, metal, alpha))
        file_dir = Path(METAL_DIR_FRAME_ALPHA.format(metal, alpha))

    else:
        raise ValueError("Alpha element abundances [alpha/M]!=0"
                         " are available for -3.0≤[Fe/H]≤0.0. only.")

    out = MODEL_SUB_DIR / file_dir /filename

    return out


def get_file(filepath, query=True):

    local_filepath = DEFAULT_MODEL_DIR / filepath
    local_filepath.parent.mkdir(parents=True, exist_ok=True)
    if local_filepath.is_file():
        log.debug(f'Reading local file: {local_filepath}')
    elif query:
        url_link = URL_ROOT + str(filepath)
        log.info(f'Downloading file: {url_link}')
        download_ftp_file(url_link, local_filepath)
        log.info(f'File saved at: {local_filepath}')
    else:
        msg = f'Cannot find {local_filepath}.  '
        msg += f'Use `query=True` to download model file.'
        log.critical(msg)
        raise FileNotFoundError(msg)

    if not local_filepath.is_file():
        log.warning(f'Failed to download {url_link}')

    return local_filepath


def get_phoenix_wv_grid(query=True):

    wv_file = Path(WAVE_FILENAME)
    out = get_file(wv_file, query=query)

    return out


def get_from_phoenix_grid(param_grid:Dict[str, list], query=True):

    # Put the param_grid keys
    param_keys = list(param_grid.keys())

    filepath_dict = dict()
    # Iterate over all possible combinations of parameters
    for params in product(*param_grid.values()):
        kwargs = {key: value for key, value in zip(param_keys, params)}
        filepath = get_phoenix_filepath(**kwargs)

        # Save full filepath value in dict
        key = tuple(kwargs.items())
        filepath_dict[key] = get_file(filepath, query=query)

    return filepath_dict


def get_phoenix_files(teff:Union[list, tuple, float] = 6000,
                     logg:Union[list, tuple, float] = 4.5,
                     metal:Union[list, tuple, float] = 0.0,
                     alpha:Union[list, tuple, float] = 0.0,
                     extrapolate:bool = False,
                     query:bool = True):

    kwargs = dict(teff=teff, logg=logg, metal=metal, alpha=alpha, extrapolate=extrapolate)
    phoenix_grid = get_phoenix_grid_in_range(**kwargs)

    wave_file = get_phoenix_wv_grid()
    flux_file = get_from_phoenix_grid(phoenix_grid, query=query)

    return wave_file, flux_file


def convert_phoenix_at_resolution(wave, flux, resolution, wv_range, oversampling=2, n_fwhm=7, output_wv_grid=None):
    # Build grid if not given, based on resolution and oversampling
    if output_wv_grid is None:
        grid_sampling = oversampling * resolution
        output_wv_grid = get_wv_constant_res(wv_range=wv_range, resolution=grid_sampling)
    # Initialize output flux
    output_flux = np.zeros_like(output_wv_grid)

    # Iterate over the wavelength ranges with different resolutions
    for phnx_rng, phnx_res in PHOENIX_RESOLUTION.items():

        is_out_of_range = (phnx_rng[0] >= wv_range[-1]) or (phnx_rng[-1] <= wv_range[0])

        if not is_out_of_range:
            # Get the most restrictive boundaries
            wv_min = np.max([phnx_rng[0], wv_range[0]])
            wv_max = np.min([phnx_rng[-1], wv_range[-1]])

            # Add pad around to ensure valid convolution
            pad_min = wv_min / resolution * n_fwhm
            pad_max = wv_max / resolution * n_fwhm

            # Resample to ensure sampling at constant resolution (R = delta_lambda / lambda)
            fkwargs = dict(wv_range=[wv_min - pad_min, wv_max + pad_max], resolution=phnx_res)
            wv_resampled, flux_resampled = resample_constant_res(wave, flux, **fkwargs)

            # Convolve with gaussian kernel and interpolate
            wv_conv, flux_conv = gauss_convolve(wv_resampled, flux_resampled, resolution, n_fwhm=7)
            fct_flux = interp1d(wv_conv, flux_conv, kind='cubic')

            # Add values to the output flux
            is_wv_in_range = (phnx_rng[0] <= output_wv_grid) & (output_wv_grid < phnx_rng[-1])
            output_flux[is_wv_in_range] = fct_flux(output_wv_grid[is_wv_in_range])

    return output_wv_grid, output_flux


def interp_phoenix_grid(teff=7500, logg=4.5, metal=0.0, alpha=0.0, wv_range=(1, 2.5),
                        resolution=70000, oversampling=2, n_fwhm=7, output_wv_grid=None,
                        method='linear', query=True):
    # Get phoenix grid covered by the specified ranges
    param_grids = get_phoenix_grid_in_range(teff=teff, logg=logg, metal=metal, alpha=alpha)

    # Check if all files are downloaded if query is False
    if not query:
        _ = get_from_phoenix_grid(param_grids, query=query)

    # Use list of keys to make sure the order is respected
    key_list = list(param_grids.keys())
    grids_length = [len(param_grids[key]) for key in key_list]
    param_grid_idx = [np.arange(length) for length in grids_length]

    # Do not consider fixed parameters as an interpolation axis
    valid_p_idx = [i for i, length in enumerate(grids_length) if length > 1]
    out_grid_axis_key = [key_list[i] for i in valid_p_idx]
    out_grid_dims = [grids_length[i] for i in valid_p_idx]

    # Initialize outputs
    output_flux_grid = None
    native_wv_grid = None
    output_data = []
    # Init param with the first values for each parameter
    # pbar = tqdm(total=math.prod(grids_length), desc="Processing")
    for p_idx in product(*param_grid_idx):
        param = {key: param_grids[key][idx] for key, idx in zip(key_list, p_idx)}
        print(param)
        wave_file, flux_file_dict = get_phoenix_files(**param)

        if native_wv_grid is None:
            hdu = fits.open(wave_file)
            native_wv_grid = hdu[0].data / 1e4  # Angstrom to microns

        # flux_file_ict has only one value, so take the first.
        hdu = fits.open(list(flux_file_dict.values())[0])
        native_flux = hdu[0].data

        # Put model at user-specified resolution and wv_range
        args = (native_wv_grid, native_flux, resolution, wv_range)
        kwargs = dict(oversampling=oversampling, n_fwhm=n_fwhm, output_wv_grid=output_wv_grid)
        output_wv_grid, flux_conv = convert_phoenix_at_resolution(*args, **kwargs)

        # # Initialize output grid
        # if output_flux_grid is None:
        #     out_grid_dims.append(len(output_wv_grid))  # Add wavelength axis
        #     output_flux_grid = np.zeros(shape=(out_grid_dims))

        # idx = [p_idx[i] for i in valid_p_idx]
        # if output_flux_grid[tuple(idx)].sum() != 0:
        #     log.info(idx)
        #     log.info([key_list[i] for i in valid_p_idx])
        # output_flux_grid[tuple(idx)] = flux_conv
        param['Spectrum'] = flux_conv
        output_data.append(param)

    #     pbar.update(1)
    # pbar.close()   

    # interp_input = [param_grids[key] for key in out_grid_axis_key]
    # interp_input.append(output_wv_grid)
    # phoenix_interp = RegularGridInterpolator(interp_input, output_flux_grid, method=method)

    return output_wv_grid, output_data #phoenix_interp, out_grid_axis_key


class PhoenixInterpGrid:

    def __init__(self, teff=7500, logg=4.5, metal=0.0, alpha=0.0, wv_range=(1, 2.5), query=True,
                 resolution=70000, oversampling=2, n_fwhm=7, output_wv_grid=None, method='cubic'):

        output = interp_phoenix_grid(teff=teff, logg=logg, metal=metal,
                                     alpha=alpha,
                                     query=query,
                                     wv_range=wv_range, resolution=resolution, oversampling=oversampling, n_fwhm=n_fwhm,
                                     output_wv_grid=output_wv_grid, method=method)

        # self.fct_interp = output[0]
        # self.parameters = output[1]
        self.wgrid = output[0]
        df = pd.DataFrame(output[1])
        self.flux = df

        # self.n_parameters = len(self.parameters)

    def __call__(self, wv, **params):

        # required_keys = self.parameters
        # try:
        #     fct_params = [params.pop(key) for key in required_keys]
        # except KeyError:
        #     msg = f"All parameters {required_keys} must be specified in `param`."
        #     raise ValueError(msg)

        # # Check if there are any remaining parameters
        # if params:
        #     msg = f"{list(params.keys())} are not valid keyword arguments. "
        #     msg += f"Here are the required ones: {required_keys}"
        #     raise TypeError(msg)

        # # Repeat the parameters for all wavelengths
        # n_wv = len(wv)
        # n_p = self.n_parameters
        # param_repeat = np.broadcast_to(fct_params, (n_wv, n_p))
        # fct_input = np.append(param_repeat, wv[:, None], axis=1)

        # out = self.fct_interp(fct_input)

        return 