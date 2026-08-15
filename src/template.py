import numpy as np
import pandas as pd
import torch
import pickle
from transformer import *
from convolution import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Template:
    """ This class has been updated for work with real data. Real data has a caveat that 
    different observations will have different wavelengths grid. This means that we need to loop over each
    observation to interpolate it from its own wavelength grid to the upsampled_wgrid (PHOENIX grid)
    """

    def __init__(self,obs_temp,obs_berv,inst_wgrid,upsampled_wgrid,obs_wgrids=None,sys_vel=None):
        '''
        Here we initialize the template object. We pass it the observations used to make the template,
        the bervs of those observations, and the instrument wgrid. I also provide
        the upsampled wgrid that will be used as the wgrid for the template.

        REAL DATA UPDATE - If obs_wgrid is provided, this will loop over each of the observations and interpolate
        to the upsampled_wgrid which is the PHOENIX grid.

        REAL DATA UPDATE - also added an explicit, optional sys_vel parameter. It is up to your discretion about whether or not you want to include 
        the systemic velocity shift in the Template. I have chosen to do so in the current version of the pipeline with Barnard's star 
        '''
        self.observations_for_template = obs_temp     #flux of observations
        self.inst_wgrid = inst_wgrid                  #instrument wavelength grid (not used if obs_wgrids provided)
        self.upsampled_wgrid = upsampled_wgrid        #grid we are upsampling to
        self.obs_berv = obs_berv                      #BERVs
        self.obs_wgrids = obs_wgrids                  #per-observation wavelength grids, if they are provided. 
        self.sys_vel = sys_vel                        #systemic velocity shift, if provided
        self.N = len(obs_berv)                        # num of obs. Needed to know how many observations to loop through and interpolate from their wavelength grid

    def make_template(self, func='scipy'):
        ''' Here we actually make the template. You have to specify which interpolation function you would
        like to use. 
        REAL DATA UPDATE - If the obs_wgrids are provided, we will loop over each observation and interpolate to the upsampled_wgrid (PHOENIX grid).
        If the obs_wgrid is not provided, we will assume the observations are on the same grid and just interpolate from the instrument grid (inst_wgrid) to the upsampled_wgrid.
        '''
        # Loop through each observation at a time, necessary for each observation's unique wavelength grid and to allow scipy in interpolate()
        # to handle all the observations correctly.  Scipy is my chosen function as it is more accurate than connors.
        if self.obs_wgrids is not None:
            upsampled_obs_list = []
            for i in range(self.N):
                source_grid = self.obs_wgrids[i].view(1, 1, -1).to(DEVICE)  #wavelength grid of particular observation
                source_flux = self.observations_for_template[:, i:i+1, :]   #associated flux of observation
                target_grid = self.upsampled_wgrid.view(1, 1, -1).to(DEVICE) #grid we are upsampling to
               
                #this performs the interpolation                            
                interp = interpolate(source_grid, source_flux, target_grid, func)  # I am using scipy interpolation

                # Connors will return 3D but scipy 2D. Add extra dimension if needed
                if interp.dim() == 2:
                    interp = interp.unsqueeze(1)
                upsampled_obs_list.append(interp)

            upsampled_observations = torch.cat(upsampled_obs_list, dim=1)  # [1, N, M]

        else:   #synthetic observations are already on the same grid, use a batched call
            spec_wgrid_batched = self.upsampled_wgrid.view(1, 1, -1).expand(1, self.N, -1)
            inst_wgrid_batched = self.inst_wgrid.view(1, 1, -1).expand(1, self.N, -1)
            upsampled_observations = interpolate(inst_wgrid_batched, self.observations_for_template, spec_wgrid_batched, func)

        # Add extra dimension if needed, this depends on your choice of interpolation function    
        if upsampled_observations.dim() == 2:
            upsampled_observations = upsampled_observations.unsqueeze(0)

        # REAL DATA UPDATE - subtract sys_vel here internally if it was provided. This is up to the caller to decide if sys_vel is to be passed
        if self.sys_vel is not None:
            total_shift = self.obs_berv - self.sys_vel
        else:
            total_shift = self.obs_berv

        # Shift them by the total desired shift
        self.berv_shifted_observations = torch.zeros_like(upsampled_observations).to(DEVICE)
        for i in range(self.N):
            self.berv_shifted_observations[:, i, :] = shift_spectrum(upsampled_observations[:, i:i+1, :],
                -total_shift[i].unsqueeze(0).unsqueeze(0), self.upsampled_wgrid,func).squeeze(0)

        # Take the median across the observation (dim=1) and ignore NaNs
        self.template = torch.nanmedian(self.berv_shifted_observations, dim=1)[0].squeeze()
        
        return self.template