import numpy as np
import pandas as pd
import torch 
import pickle
from transformer import *
from convolution import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Template:

    def __init__(self,obs_temp,obs_berv,inst_wgrid,upsampled_wgrid) -> None:
        '''
        Here we initialiez the template object. We pass it the observations used to make the template,
        the bervs of those observations, and the instrument wgrid. I also provide
        the upsampled wgrid that will be used as the wgrid for the template.
        '''

        self.observations_for_template = obs_temp
        self.inst_wgrid = inst_wgrid
        self.upsampled_wgrid = upsampled_wgrid
        self.obs_berv = obs_berv
        print(self.obs_berv.shape)
        pass

    def make_template(self,func='scipy'):
        '''
        Here we actually make the template. You have to specify which interoplation function you would
        like to use. 
        '''
        # Upsample the observations to the upsampled wgrid

        spec_wgrid_batched = self.upsampled_wgrid.view(1, 1, len(self.upsampled_wgrid)).expand(1, len(self.obs_berv), len(self.upsampled_wgrid)).to(DEVICE)
        inst_wgrid_batched = self.inst_wgrid.view(1, 1, len(self.inst_wgrid)).expand(1, len(self.obs_berv), len(self.inst_wgrid)).to(DEVICE)
        upsampled_observations = interpolate(inst_wgrid_batched,self.observations_for_template,spec_wgrid_batched,func)
        # Shift them by BERV 
        self.berv_shifted_observations = torch.zeros_like(upsampled_observations).to(DEVICE)
        for i in range(len(self.berv_shifted_observations)):
            self.berv_shifted_observations[i] = shift_spectrum(upsampled_observations[i].view(1, 1, -1),-self.obs_berv[i].unsqueeze(0).unsqueeze(0),self.upsampled_wgrid.unsqueeze(0).unsqueeze(0))

        # Take the median 
        self.template =torch.median(self.berv_shifted_observations,axis=0)[0]

        return self.template