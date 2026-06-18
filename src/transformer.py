import torch 
import torch.nn as nn
from astropy import constants as const
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import argrelextrema
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64) 
import time

def forward_model(x_unpad,spec_wgrid_trimmed,inst_wgrid,berv,V, sys_vel = None):
    '''
    This defines the forward model so how we transform our spectrum parameter into our our spectra that resemble our observations.

    x_unpad is the spectrum samples without their padding 
    spec_wgrid_trimmed is the native wavelength grid of the spectrum parameter without the padding
    inst_wgrid is the wavelength grid of our observations (this is assumed to be the same across our observations)
    berv is the berv value for each observation
    V is the suggested RV value for each observation
    REAL DATA UPDATE: systemic velocity for each observation. If None then assumed to be zero.

    so we go from [B,D] to [B,N,L] where
    B = the batch size of spectrum samples
    D = is the length of your spectrum parameter
    N = the number of observations 
    L = the length of your spectrum in observation space (instrument pixels)
    '''
    # This is the batch size, so the number of spectrum samples we need to transform
    B = len(x_unpad)

    if sys_vel is not None:
        total_vel = berv + V + sys_vel
    else:
        total_vel = berv + V

    # Shift and interpolate to match observations
    # First we need to batch the spectrum wavelength grid
    spec_wgrid_batched = spec_wgrid_trimmed.view(1, 1, len(spec_wgrid_trimmed)).expand(B, len(V[0]),len(spec_wgrid_trimmed))

    # Then we apply a doppler shift to our spectrum samples according to berv and V
    # We get back the flux values with respect to the orginal spectrum wavelength grid
    shifted_obs = shift_spectrum(x_unpad,total_vel,spec_wgrid_batched)

    # Then we batch the observation wavelength grid
    inst_wgrid_batched = inst_wgrid.view(1, 1, len(inst_wgrid)).expand(B, len(V[0]), len(inst_wgrid))

    # Finally we interpolate our shifted spectra to our observations wavelength grid
    transformed_X = interpolate(spec_wgrid_batched,shifted_obs,inst_wgrid_batched)

    return transformed_X

def shift_spectrum(S: torch.Tensor, V: torch.Tensor, W: torch.Tensor,func='connors') -> torch.Tensor:
        '''
        S is the rest frame stellar spectrum given as torch tensor [B,1,len(wgrid)]
        V is a vector of N velocities (must be in m/s) given as torch tensor [B,N] N=number of observations
        W is the native wgrid

        return shifted_S which is S shifted at the V velocities, it will be a 2D tensor
        '''
        V = V.to(DEVICE)
        if V.ndim==0:
            V = torch.tensor([V],device=DEVICE)

        # Reshape tensors
        B = S.shape[0]
        # batched_wgrid = W.view(1, 1, len(W)).expand(B, len(V[0]), len(W))
        V = V.unsqueeze(-1)
        S = S.expand(B, len(V[0]), -1)
        speed_of_light_ms = const.c.value
        # relativistic calculation (1 - v/c)
        part1 = 1 - (V / speed_of_light_ms)
        # relativistic calculation (1 + v/c)
        part2 = 1 + (V / speed_of_light_ms)

        shifted_grid = W * torch.sqrt(part1 / part2)
        shifted_S = interpolate(shifted_grid,S,W,func)
        return shifted_S

def interpolate(x, y, xs, func='connors'):
    ''' Function to interpolate from grid x to grid xs, where y is the value on grid x. 
    For the scipy function, the input can be 3D, but they neeed to be squeezed to 2D [N, L]]
    '''
    if func == 'scipy':
        #the following will remove any leading dimesions of size 1 until we get to the [N, L] shape that scipy expects.
        # we want to make sure to keep the last dimension which is the wavelength dimension.
        while x.dim() > 2 and x.size(0) == 1:
            x = x.squeeze(0)
        while y.dim() > 2 and y.size(0) == 1:
            y = y.squeeze(0)
        while xs.dim() > 2 and xs.size(0) == 1:
            xs = xs.squeeze(0)

        # Now ensure they are at least 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if xs.dim() == 1:
            xs = xs.unsqueeze(0)
        return scipys(x, y, xs)
    elif func == 'connors':
        return connors(x, y, xs)



### SCIPY INTERPOLATION FUNCTION ########
def scipys(x,y,xs):
    """
    Interpolate a 2D tensor of spectra using scipy's InterpolatedUnivariateSpline.

    This function has now been updated to processes a batch of spectra (rows of `x` and `y`) and
    interpolates each onto a new grid `xs`. 

    Parameters:
    x (torch.Tensor): Original wavelengths for each of N spectra.
    y (torch.Tensor): Original fluxes for each spectrum.
    xs (torch.Tensor): Target wavelengths (new grid) for each spectrum. These are the PHOENIX grids

    Returns
    -------
    ys (torch.Tensor): Interpolated fluxes on the new grid `xs` for each spectrum. 
    """
    N, L = x.shape  #N, is the number of spectra, L is wavelengths points in the original grid
    M = xs.shape[1]  #target grid number of points
    ys = torch.zeros(N, M, device=DEVICE, dtype=torch.float64) # Initialize the output tensor of the same shape as xs

    # We are now going to loop through each spectrum individually
    for i in range(N):
        #convert to numpy first. We will need to remove any NaNs before interpolation
        x_np = x[i].cpu().numpy().astype(np.float64)
        y_np = y[i].cpu().numpy().astype(np.float64)
        xs_np = xs[i].cpu().numpy().astype(np.float64)

        # Remove NaNs. We get a mask and keep only those in the interpolation
        valid = ~np.isnan(x_np) & ~np.isnan(y_np)

        #get only the valid points in wavelengths and in flux
        x_clean = x_np[valid]
        y_clean = y_np[valid]

        # For each spectrum (row), create an InterpolatedUnivariateSpline instance
        spline = InterpolatedUnivariateSpline(x_clean, y_clean, k=3, ext=1)
        # Interpolate to get the new y values at xs[i]
        ys[i] = torch.tensor(spline(xs_np), dtype=torch.float64).to(DEVICE)
    return ys


def connors(x, y, xs, extend='const'):
    """
    Interpolate spectra using Connor's Splining Code for batched and multi-grid inputs.

    Parameters:
    - x: [B, N, L] — original x values
    - y: [B, N, L] — original y values
    - xs: [B, N, M] — new x values to interpolate onto

    Returns:
    - ys: [B, N, M] — interpolated y values
    """
    B, N, L = x.shape
    _, _, M = xs.shape  # M may not equal L

    # Compute Hermite slopes: [B, N, L-1]
    delta_x = x[..., 1:] - x[..., :-1]
    delta_y = y[..., 1:] - y[..., :-1]
    m = delta_y / delta_x  # [B, N, L-1]

    # Adjust to [B, N, L] using Hermite rule
    m = torch.cat([
        m[..., [0]], 
        (m[..., 1:] + m[..., :-1]) / 2, 
        m[..., [-1]]
    ], dim=-1)  # [B, N, L]

    # Flatten batch for searchsorted (works only on 2D)
    x_flat = x.reshape(-1, L)      # [(B*N), L]
    xs_flat = xs.reshape(-1, M)    # [(B*N), M]

    # Get interpolation indices
    idxs = torch.searchsorted(x_flat[:, :-1].contiguous(), xs_flat.contiguous(), right=True) - 1  # [(B*N), M]
    idxs = idxs.clamp(min=0, max=L - 2)
    idxs = idxs.view(B, N, M)  # [B, N, M]

    # Utility to gather from [B, N, L] using [B, N, M] indices
    def batched_gather(tensor, idx):
        B, N, L = tensor.shape
        _, _, M = idx.shape
        # Expand indices for batch/grid dims
        batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(B, N, M)
        grid_idx = torch.arange(N, device=idx.device).view(1, N, 1).expand(B, N, M)
        return tensor[batch_idx, grid_idx, idx]  # [B, N, M]

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
    hh = _h_poly(s)  # [4, B, N, M]

    # Interpolated result
    ret = (
        hh[0] * y0 + hh[1] * m0 * dx + hh[2] * y1 + hh[3] * m1 * dx
    )  # [B, N, M]

    # Handle extrapolation
    x_last = x[..., -1:]  # [B, N, 1]
    y_last = y[..., -1:]  # [B, N, 1]
    x_last = x_last.expand(-1, -1, M)
    y_last = y_last.expand(-1, -1, M)

    if extend == "const":
        indices = xs > x_last
        ys = torch.where(indices, y_last, ret)

    elif extend == "linear":
        x_prev = x[..., -2:-1].expand(-1, -1, M)
        y_prev = y[..., -2:-1].expand(-1, -1, M)
        slope = (y_last - y_prev) / (x_last - x_prev)
        indices = xs > x_last
        ys = torch.where(indices, y_last + (xs - x_last) * slope, ret)
    else:
        ys = ret  # default if extend is not specified

    return ys


def _h_poly(s):
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return torch.stack([h00, h10, h01, h11], dim=0)  # [4, B, N, M]