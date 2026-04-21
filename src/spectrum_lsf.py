from torch.func import grad
from torch import vmap # parallelize a function over its batch dimension
import torch
import numpy as np
from transformer import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
from types import SimpleNamespace
def Score_Likelihood(Y: torch.Tensor,V: torch.Tensor,sig_n: float,berv, spec_wgrid, inst_wgrid, non_ones,SNR, beta_min: float,
                    beta_max: float,AtA: torch.Tensor):
    '''
    This is the score likelihood function class. It's inputs are the set of observations and parameters
    that will be used to define the likelihood score. This is function used to compute the posterior sample
    of the spectrum, keeping the velocities fixed. This uses the Convolved Likelihood for Variance Preserving SDE shown in Noe's Paper

    INPUTS:
    Y: Observations given by a torch tensor of [1, N, L] where N is num of observations and L is length of spectrum (detector)
    V: Vector of velocities given by a torch tensor of [N]
    sig_n: The sqrt(std) of the gaussian noise added to the observations is [1, N, L]

    berv: observations berv
    spec_wgrid: the wavelength grid of the continous spectrum
    inst_wgrid: the wavelength grid of the observation
    non_ones: this is to unpad the wavelength grid
    SNR: snr of observations

    beta_min: the beta_min used to train the model
    beta_max: the beta_max used to train the model

    AAT: the A matrix to trasnform the uncertainty in diffusion model through the transformation of the sample

    OUTPUTS:
    score_llk: This returns the function that score_models can use to do posterior sampling
    '''

    def find_sigma_t(t):
        
        beta_primitive = 0.5 * (beta_max - beta_min) * t**2 + beta_min * t
        mu = torch.exp(-0.5 * beta_primitive)
        std = (1 - mu ** 2).sqrt()
        
        return std, mu


    def find_Sigma(sigma_t,mu,B,N,L,D):
        '''
        Calcuates the uncertainty matrix used for likelihood calculation
        '''
        ## FASTER VERSION
        # Precompute A A^T once (independent of batch!)
        # shape [N, L, L]
        # Scale by sigma_t^2 per batch
        sig_AAt = (sigma_t**2).view(B, 1, 1,1) * AtA.unsqueeze(0)  # [B, N, L, L]
        # Diagonal mu^2 * sig_n^2 term
        sig_mat = (mu**2).view(B, 1, 1, 1) * torch.diag_embed(sig_n**2).expand(1,N, -1,-1)  # [B,1,L,L]
        # sig_mat = sig_mat.expand(B, N, L, L)  
        sig_factor = 1
        Sigma = sig_AAt*sig_factor + sig_mat
        del sig_AAt, sig_mat

        return Sigma

    def cholesky_fast(y, mu, x, sig):
        """
        This calculates the likelihood in an efficient way when we have large matrices to invert
        y: [1, N, L]
        x: [B, N, L]
        sig: [B, N, L, L]
        """
    
        B, N, L_full = x.shape

        # We will clip the ends because these get messed up from interpolation errors
        start = int(0.005 * L_full)
        end = int(0.995 * L_full)
        
        x_clip = x[:, :, start:end]                 # [B, N, L]
        y_clip = y[:, :, start:end] * mu[:, None, None]  # Broadcast instead of repeat
        sig_clip = sig[:, :, start:end, start:end]  # [B, N, L, L]

        # Calculate the residual between data and forward model 
        resid = y_clip - x_clip                     # [B, N, L]
        resid = resid.unsqueeze(-1)                 # [B, N, L, 1]
    
        # Cholesky decomposition (batch)
        L_chol = torch.linalg.cholesky(sig_clip)    # [B, N, L, L]
    
        # Solve triangular system: L @ z = resid
        z = torch.linalg.solve_triangular(L_chol, resid, upper=False)  # Faster than cholesky_solve for batches
    
        # Mahalanobis term
        quad = torch.sum(z**2, dim=-2).squeeze(-1)  # [B, N], z^T z = (L⁻¹ resid)^T * (L⁻¹ resid)
    
        # Log determinant
        logdet = 2 * torch.sum(torch.log(torch.diagonal(L_chol, dim1=-2, dim2=-1)), dim=-1)  # [B, N]
    
        const = (end - start) * np.log(2 * np.pi)
    
        llk = -0.5 * (quad + logdet + const)
        return llk


    def likelihood_fn(t,x):
        '''
        This calculates the likelihood using the convolved likelihood approximation of our spectrum sample
        with respect to our observations

        t is in shape [B] where B is num samples
        x is in shape [B,1,D] where B is num of samples and D is length of spectrum (upsampled)
        '''

        # Dimension
        B = len(x)
 
        x_unpad = x[:,:,non_ones[0] : non_ones[-1] + 1]
        D = x_unpad.shape[-1]
        L = Y.shape[-1]
        N = Y.shape[1]

        # Calculate the sigmas - this is to calculate the uncertainty matrix in our llk calculation
        sigma_t, mu = find_sigma_t(t) # output is [B]
        Sigma = find_Sigma(sigma_t,mu, B,N,L,D) # output is [B,N,L,L]

        #### Transform the diffusion model output ####
        transformed_X = forward_model(x_unpad,spec_wgrid,inst_wgrid,berv,V)

        # Calculate likelihood with transformed x
        llk = cholesky_fast(Y,mu,transformed_X,Sigma)

        return llk.sum()

    def score_llk(t, x):
        '''
        This function recieves t, x and outputs the score of the likelihood function with respect to x
        t = noise parameter (time)
        x = spectrum sample (batched)
        '''
        score = grad(likelihood_fn,argnums=1)(t,x)
        return score
        
    return score_llk