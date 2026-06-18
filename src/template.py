import numpy as np
import pandas as pd
import torch
import pickle
from transformer import *
from convolution import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Template:
    """ This class has been updated for work with real data. Real data has a caveat that 
    different observations have different wavelengths grid. This means that we need to loop over each
    observation from its own wavelength grid to the upsampled_wgrid (PHOENIX grid)
    """

    def __init__(self,obs_temp,obs_berv,inst_wgrid,upsampled_wgrid,obs_wgrids=None):
        '''
        Here we initialize the template object. We pass it the observations used to make the template,
        the bervs of those observations, and the instrument wgrid. I also provide
        the upsampled wgrid that will be used as the wgrid for the template.

        If obs_wgrid is provided, this will loop over each of the observations and interpolate
        to the upsampled_wgrid which is the PHOENIX grid.
        '''

        self.observations_for_template = obs_temp  #flux of observations
        self.inst_wgrid = inst_wgrid               #instrument wavelength grid (not used if obs_wgrids provided)
        self.upsampled_wgrid = upsampled_wgrid    #grid we are upsampling to
        self.obs_berv = obs_berv                  #BERVs
        self.obs_wgrids = obs_wgrids  # this is the per-observation wavelength grids, if they are provided. 
                                      #  If not, we will assume the observations are already on the same grid as the upsampled_wgrid and just interpolate from the instrument grid (inst_wgrid) to the upsampled_wgrid.
        self.N = len(obs_berv)  #this will keep track of the number of obseravtions

    def make_template(self, func='scipy'):
        ''' Here we are creating the template.
        If the obs_wgrids are provided, we will loop over each observation and interpolate to the upsampled_wgrid (PHOENIX grid).
        If the obs_wgrid is not provided, we will assume the observations are already on the same grid as the upsampled_wgrid and just interpolate from the instrument grid (inst_wgrid) to the upsampled_wgrid.
        '''
        if self.obs_wgrids is not None:
            upsampled_obs_list = []
            for i in range(self.N):
                src_grid = self.obs_wgrids[i].view(1, 1, -1).to(DEVICE)  #wavelength grid of particular observation
                src_flux = self.observations_for_template[:, i:i+1, :]   #associated flux of observation
                tgt_grid = self.upsampled_wgrid.view(1, 1, -1).to(DEVICE) #grid we are upsampling to
               
                #this performs the interpolation 
                interp = interpolate(src_grid, src_flux, tgt_grid, func)

                upsampled_obs_list.append(interp.unsqueeze(1))
            upsampled_observations = torch.cat(upsampled_obs_list, dim=1)  # [1, N, M]

        else:   #synthetic observations are already on the same grid
            spec_wgrid_batched = self.upsampled_wgrid.view(1, 1, -1).expand(1, self.N, -1)
            inst_wgrid_batched = self.inst_wgrid.view(1, 1, -1).expand(1, self.N, -1)
            upsampled_observations = interpolate(inst_wgrid_batched, self.observations_for_template, spec_wgrid_batched, func)
        
        if upsampled_observations.dim() == 2:
            # If it's [N, L], add batch dimension -> [1, N, L]
            upsampled_observations = upsampled_observations.unsqueeze(0)

        elif upsampled_observations.dim() == 3:
            pass  # Already has batch dimension
        
        # Shift them by BERV
        self.berv_shifted_observations = torch.zeros_like(upsampled_observations).to(DEVICE)
        for i in range(self.N):
            self.berv_shifted_observations[:, i, :] = shift_spectrum(upsampled_observations[:, i:i+1, :],
                -self.obs_berv[i].unsqueeze(0).unsqueeze(0), self.upsampled_wgrid,func).squeeze(0)

        # Take the median and ignore NaNs (there shouldn't be NaNs in the template anyway since we masked them when doing the interpolation)
        self.template = torch.nanmedian(self.berv_shifted_observations, dim=1)[0].squeeze()
        
        return self.template