import torch 
import torch.nn as nn
from astropy import constants as const
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import argrelextrema
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64) 
import time

def forward_model(x_unpad, spec_wgrid_trimmed, inst_wgrid, berv, V, sys_vel=None, obs_wgrids=None):
    '''
    This defines the forward model so how we transform our spectrum parameter into our spectra that resemble our observations.

    x_unpad is the spectrum samples without their padding 
    spec_wgrid_trimmed is the native wavelength grid of the spectrum parameter without the padding
    inst_wgrid is the wavelength grid of our observations (this is assumed to be the same across our observations for synthetic data)
    REAL DATA UPDATE - obs_wgrid: per observation wavelength grids. 
    REAL DATA UPDATE - sys_vel: Updated to be able to take in a systemic velocity now
    berv is the berv value for each observation
    V is the suggested RV value for each observation

    so we go from [B,D] to [B,N,L] where
    B = the batch size of spectrum samples
    D = is the length of your spectrum parameter
    N = the number of observations 
    L = the length of your spectrum in observation space (instrument pixels)
    '''
    # The spectrum can arrive with or without the channel axis -- the
    # diffusion model gives [B, 1, D], MALA passes the posterior as [B, D]
    if x_unpad.dim() == 2:
        x_unpad = x_unpad.unsqueeze(1)

    B = x_unpad.shape[0] # Number of spectra samples

    # The original used len(V[0]) throughout, which fails if V comes in
    # as a single scalar velocity rather than a batch
    if V.dim() == 0:
        N = 1
    else:
        N = V.shape[1]  # Number of observations

    # REAL DATA UPDDATE - Fixed to match our convention we are using in the real data code
    if sys_vel is not None:
        total_vel = berv - sys_vel - V
    else:
        total_vel = berv - V

    # REAL DATA UPDATE - which grid to degrade onto. The original assumed one
    # shared inst_wgrid; real observations each have their own.
    if obs_wgrids is not None:
        inst_wgrids = obs_wgrids
    else:
        if inst_wgrid.dim() == 1:
            inst_wgrids = inst_wgrid.unsqueeze(0) # synthetic: one shared grid
        else:
            inst_wgrids = inst_wgrid  # already 2D - from MALA

    # One grid per observation
    if inst_wgrids.shape[0] != N:
        inst_wgrids = inst_wgrids.expand(N, -1)

    # Shift and interpolate to match observations
    # First we need to batch the spectrum wavelength grid
    spec_wgrid_batched = spec_wgrid_trimmed.view(1, 1, -1).expand(B, N, -1)

    # Then we apply a doppler shift to our spectrum samples according to berv and V
    # We get back the flux values with respect to the orginal spectrum wavelength grid
    shifted_obs = shift_spectrum(x_unpad, total_vel, spec_wgrid_batched)

    # Then we batch the observation wavelength grid
    inst_wgrid_batched = inst_wgrids.unsqueeze(0).expand(B, N, -1)

    # Finally we interpolate our shifted spectra to our observations wavelength grid
    transformed_X = interpolate(spec_wgrid_batched, shifted_obs, inst_wgrid_batched)

    return transformed_X

def shift_spectrum(S: torch.Tensor, V: torch.Tensor, W: torch.Tensor, func='connors') -> torch.Tensor:
    '''
    S is the rest frame stellar spectrum given as torch tensor [B,1,len(wgrid)]
    V is a vector of N velocities (must be in m/s) given as torch tensor [B,N]
    W is the native wgrid
    '''
    V = V.to(DEVICE)
    if V.ndim == 0:
        V = torch.tensor([V], device=DEVICE)

    # REAL DATA UPDATE - W has to be 3D before shifted_grid is built below
    # Template passes a 1D grid, and with func='connors' that reaches connors(), which unpacks three dimensions and fails. 
    # scipys() only reads index [0], this crash only appeared once connors was tried.
    if W.dim() == 1:
        W = W.view(1, 1, -1)

    # Reshape tensors
    B = S.shape[0]
    V = V.unsqueeze(-1)
    S = S.expand(B, len(V[0]), -1)
    speed_of_light_ms = const.c.value
    # relativistic calculation (1 - v/c)
    part1 = 1 - (V / speed_of_light_ms)
    # relativistic calculation (1 + v/c)
    part2 = 1 + (V / speed_of_light_ms)

    shifted_grid = W * torch.sqrt(part1 / part2)
    shifted_S = interpolate(shifted_grid, S, W, func)
    return shifted_S

def interpolate(x, y, xs, func='connors'):
    ''' Interpolate from grid x to grid xs. 
    REAL DATA UPDATE - scipys wants [N, L], and calls pass anything from 1D to [1, N, L]. The original just did scipys(x[0], y[0], xs[0]),
    which oreviosuly kept only the first batch element and discarded the rest
    Squeeze away any leading axes of length 1, then add one back if what is left is 1D.
    '''
    if func == 'scipy':
        # This is the most accurate as it computes second derivatives but not torch compatible 
        # or batchwise compatible
        # Use scipy InterpolatedUnivariateSpline (not batched)
        while x.dim() > 2 and x.size(0) == 1:
            x = x.squeeze(0)
        while y.dim() > 2 and y.size(0) == 1:
            y = y.squeeze(0)
        while xs.dim() > 2 and xs.size(0) == 1:
            xs = xs.squeeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)
        return scipys(x, y, xs)
    
    elif func == 'connors':
        # This is an order of magnitude less accurate but torch compatible and batchwise
        # compatible making it very fast
        # Use the batched version of connor's code 
        return connors(x, y, xs)

### SCIPY INTERPOLATION FUNCTION ########
def scipys(x, y, xs):
    """
    Interpolate a 2D tensor of spectra using scipy's InterpolatedUnivariateSpline.

    Parameters:
    - x (torch.Tensor): 2D tensor of shape [N, L] (original x values)
    - y (torch.Tensor): 2D tensor of shape [N, L] (original y values)
    - xs (torch.Tensor): 2D tensor of shape [N, M] (new x values for interpolation)

    Returns:
    - ys (torch.Tensor): 2D tensor of shape [N, M] (interpolated y values)

    REAL DATA UPDATE - NaN are dropped before fitting, since the spline cannot fit through them.
    Wavelengths outside the range the spline was fitted over are set to NaN by hand
    ext=1 returns 0.0 there, which produced a horizontal artifact
    Then those zeros were being incorrectly smoothly fitted across by the second spline during the BERV shift
    scipy has no "return NaN" option, hence doing it manually.
    """
    N, L = x.shape
    M = xs.shape[1]  # target grid length
    ys = torch.zeros(N, M, device=DEVICE, dtype=torch.float64)

    # One spectrum at a time as scipy's spline is not batched
    for i in range(N):
        x_np = x[i].cpu().numpy().astype(np.float64)
        y_np = y[i].cpu().numpy().astype(np.float64)
        xs_np = xs[i].cpu().numpy().astype(np.float64)

        # REAL DATA UPDATE - drop the NaN before fitting
        valid = ~np.isnan(x_np) & ~np.isnan(y_np)
        x_clean = x_np[valid]
        y_clean = y_np[valid]

        # For each spectrum (row), create an InterpolatedUnivariateSpline instance
        spline = InterpolatedUnivariateSpline(x_clean, y_clean, k=3, ext=1)
        result = spline(xs_np)

        # Manually mask out-of-domain queries as NaN, since ext=1 would
        # otherwise silently fill them with 0.0
        out_of_domain = (xs_np < x_clean.min()) | (xs_np > x_clean.max())
        result[out_of_domain] = np.nan

        ys[i] = torch.tensor(result, dtype=torch.float64).to(DEVICE)
    return ys


def connors(x, y, xs, extend='const'):
    """
    Interpolate spectra using Connor's Splining Code for batched and multi-grid
    inputs. Cubic Hermite, torch-native and differentiable, which is why it is
    the only option inside the likelihood and MALA.

    Parameters:
    - x: [B, N, L] — original x values
    - y: [B, N, L] — original y values
    - xs: [B, N, M] — new x values to interpolate onto

    Returns:
    - ys: [B, N, M] — interpolated y values

    REAL DATA UPDATES: 
    x and y are sliced down to the valid (non-NaN) region before any slope is computed, so the slope tensor 
    is built purely from real data and the gradient stays clean.

    extend='const' used to clamp out-of-domain queries to the nearest edge value
    It now returns genuine NaN.
    The left edge is handled too. The original only checked whether a point in xs was past the last data point, never whether it was before the first.
    The slicing only removes NaN at the ends. Interior gaps are filled by fill_internal_nans before anything gets here
    """
    B, N, L = x.shape
    _, _, M = xs.shape  # M may not equal L

    # REAL DATA UPDATE to handle the NaNs
    nan_mask = torch.isnan(y[0, 0])
    if nan_mask.any():
        # Cut down to the first and last real pixel. This only removes NaN at the edges. A NaN in the middle would survive it. fill_internal_nans
        # is what guarantees there are none.
        valid_idx = torch.where(~nan_mask)[0]
        v_start, v_end = valid_idx[0].item(), valid_idx[-1].item() + 1
        x = x[..., v_start:v_end]
        y = y[..., v_start:v_end]
        L = x.shape[-1]

    delta_x = x[..., 1:] - x[..., :-1]
    delta_y = y[..., 1:] - y[..., :-1]
    m = delta_y / delta_x

    # Adjust to [B, N, L] using Hermite rule
    m = torch.cat([
        m[..., [0]],
        (m[..., 1:] + m[..., :-1]) / 2,
        m[..., [-1]]
    ], dim=-1)

    # Flatten batch for searchsorted (works only on 2D)
    x_flat = x.reshape(-1, L)
    xs_flat = xs.reshape(-1, M)

    # Get interpolation indices
    idxs = torch.searchsorted(x_flat[:, :-1].contiguous(), xs_flat.contiguous(), right=True) - 1
    idxs = idxs.clamp(min=0, max=L - 2)
    idxs = idxs.view(B, N, M)

    # Utility to gather from [B, N, L] using [B, N, M] indices
    def batched_gather(tensor, idx):
        B, N, L = tensor.shape
        _, _, M = idx.shape
        batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(B, N, M)
        grid_idx = torch.arange(N, device=idx.device).view(1, N, 1).expand(B, N, M)
        return tensor[batch_idx, grid_idx, idx]

    # Gather required points
    x0 = batched_gather(x, idxs)
    x1 = batched_gather(x, idxs + 1)
    y0 = batched_gather(y, idxs)
    y1 = batched_gather(y, idxs + 1)
    m0 = batched_gather(m, idxs)
    m1 = batched_gather(m, idxs + 1)

    dx = x1 - x0
    s = (xs - x0) / dx

    # Hermite basis
    hh = _h_poly(s)

    # Interpolated result
    ret = (
        hh[0] * y0 + hh[1] * m0 * dx + hh[2] * y1 + hh[3] * m1 * dx
    )

    # Handle extrapolation
    x_last = x[..., -1:]
    x_last = x_last.expand(-1, -1, M)

    # REAL DATA UPDATE - x_first is new. The original only ever looked at the right-hand edge.
    x_first = x[..., :1]
    x_first = x_first.expand(-1, -1, M)

    if extend == "const":
        # REAL DATA UPDATE - return NaN instead of clamping to the edge value.
        # I don't believe we require any trimming after forward modelling. I believe changing what 
        # value we extend to has fixed this. Before I was getting horizontal vertical artifcats, but using the Nan
        # replacement should hopefully has fixed it
        indices_right = xs > x_last
        indices_left = xs < x_first
        nan_tensor = torch.full_like(ret, float('nan'))
        ys = torch.where(indices_right, nan_tensor, ret)
        ys = torch.where(indices_left, nan_tensor, ys)
    elif extend == "linear":
        # REAL DATA UPDATE - y_first and the whole left-hand half below are new,
        # for the same reason as above
        y_last = y[..., -1:].expand(-1, -1, M)
        y_first = y[..., :1].expand(-1, -1, M)
        x_prev = x[..., -2:-1].expand(-1, -1, M)
        y_prev = y[..., -2:-1].expand(-1, -1, M)
        slope_right = (y_last - y_prev) / (x_last - x_prev)
        indices_right = xs > x_last
        ys = torch.where(indices_right, y_last + (xs - x_last) * slope_right, ret)

        x_next = x[..., 1:2].expand(-1, -1, M)
        y_next = y[..., 1:2].expand(-1, -1, M)
        slope_left = (y_next - y_first) / (x_next - x_first)
        indices_left = xs < x_first
        ys = torch.where(indices_left, y_first + (xs - x_first) * slope_left, ys)
    else:
        ys = ret # default if extend is not specified

    return ys


def _h_poly(s):
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return torch.stack([h00, h10, h01, h11], dim=0)  # [4, B, N, M]


def fill_internal_nans(obs_flux, native_grid):
    ''' NEW FUNCTION
    Interpolates across the NaN gaps that sit inside an observation's coverage.
    Only interior gaps are touched. The NaN regions beyond the first and last real pixel are left alone 

    INPUTS:
    obs_flux: one observation's spectrum with NaNs still in place
    native_grid: its wavelength grid

    OUTPUTS:
    obs_flux_filled: the same array with the interior gaps filled in
    '''
    valid_mask = ~torch.isnan(obs_flux)
    valid_indices = torch.where(valid_mask)[0]

    first_valid = valid_indices[0].item()
    last_valid = valid_indices[-1].item()

    # interior means between the first and last real pixel. Anything outside
    # that is an end gap and gets left alone.
    interior_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    interior_mask[first_valid:last_valid + 1] = True

    # The pixels to fill: NaN, but with real data on both sides of them
    internal_nan_mask = torch.isnan(obs_flux) & interior_mask

    obs_flux_filled = obs_flux.clone() 

    # Skipped entirely for an observation with no interior gaps 
    if internal_nan_mask.any():
        # Fit through the real pixels inside the interior only
        valid_interior_mask = valid_mask & interior_mask
        valid_wavelengths = native_grid[valid_interior_mask]
        valid_flux = obs_flux[valid_interior_mask]

        # Evaluate over the WHOLE native grid, then keep only the gap pixels.
        # Simpler than evaluating at just the gaps, and the rest is discarded
        # on the next line anyway.
        interpolated_full = interpolate(
            valid_wavelengths.unsqueeze(0).unsqueeze(0),
            valid_flux.unsqueeze(0).unsqueeze(0),
            native_grid.unsqueeze(0).unsqueeze(0)).squeeze()
        obs_flux_filled[internal_nan_mask] = interpolated_full[internal_nan_mask]

    return obs_flux_filled