import numpy as np
import pandas as pd
import torch 
import pickle
from transformer import *
from convolution import *
from scipy.optimize import minimize_scalar
torch.set_default_dtype(torch.float64) 
import matplotlib.pyplot as plt

class RV_Retrieval():
    def __init__(self, SNR,model, upsampled_wgrid, instrument_wgrid, Ntemp, type="template", obs_wgrid = None):
        '''
        In this class we will find the relative velocity between the template and an observation, 
        this relative velocity should only be the planetary signal. 

        REAL DATA UPDATE: We will have to make the same changes to this class which were done to template class.
        We will need to keep track of the varying wavelength grid for each observation. 
        This is the obs_wgrid = observation wavelength grid parameter which will be used in new_model

        '''
        self.SNR = SNR
        self.model = model
        self.upsampled_wgrid = upsampled_wgrid
        self.instrument_wgrid = instrument_wgrid
        self.Ntemp = Ntemp
        self.type = type

        #store the per-observation wavelength grids
        self.obs_wgrid = obs_wgrid
        

        pass

    def new_model(self, dv, berv,func, i=None):
        '''
        REAL OBSERVATION UPDATE: We need to account for per observation wavelength grids.
        Need to loop through observations so using an observation index i

        This function is used to shift the template by various dv values that represent 
        different planetary signals. This shifted model will then be compared to the true observation
        signal to estimate the likilihood of the RV value

        INPUTS:
        dv: an array of radial velocities that encompass all values that planetary signal might be 
        i: index of our observation
        
        OUTPUTS:
        shifted_model: an array of the template shifted by the various dv values
        '''
        # if not isinstance(dv, float):
        #     dv = dv[0]
 
        rv = dv+berv
        if rv.ndim==0:
            rv = torch.tensor([rv]).to(DEVICE)
        if self.type=='template':
            # If we are using this RV analysis technique with a template
            shifted = shift_spectrum(self.model.view(1, 1, -1),rv.unsqueeze(0),self.upsampled_wgrid.unsqueeze(0).unsqueeze(0),func)

            # Here we choose which instrument grid to use
            if self.obs_wgrid is not None and i is not None: #if we passed our varying observational wavelength grids
                inst_grid = self.obs_wgrid[i].to(DEVICE)
            else:
                inst_grid = self.instrument_wgrid
            # degrade the template to the chosen wavelength grid
            batched_wgrid = self.upsampled_wgrid.unsqueeze(0).unsqueeze(0)#.repeat(self.broadened_observations.shape[0],1).to(DEVICE)
            batched_instwgrid = inst_grid.unsqueeze(0).unsqueeze(0)
            shifted_degraded_template = interpolate(batched_wgrid,shifted,batched_instwgrid,func)


        elif self.type=="sample":
            # If we are using this RV analysis technique with a posterior sample or intrinsic spectrum
            right_flux = self.model
            shifted = shift_spectrum(right_flux.view(1, 1, -1),rv.unsqueeze(0),self.upsampled_wgrid.unsqueeze(0).unsqueeze(0),func)
            if self.obs_wgrid is not None and i is not None:
                inst_grid = self.obs_wgrid[i].to(DEVICE)
            else:
                inst_grid = self.instrument_wgrid
            batched_wgrid = self.upsampled_wgrid.unsqueeze(0).unsqueeze(0)#.repeat(self.broadened_observations.shape[0],1).to(DEVICE)
            batched_instwgrid = inst_grid.unsqueeze(0).unsqueeze(0)
            shifted_degraded_template = interpolate(batched_wgrid,shifted,batched_instwgrid,func)


        return shifted_degraded_template.cpu()

    def chi2(self,v,data,sig,berv,func,i=None):
        '''
        This function estimates the chi2 of various dv values for the observation provided
        based on Equation 2 of (Silva et al. 2022).

        INPUTS:
        data: the observation spectrum
        v: the list of dv values to test
        sig: the sigma of the gaussian noise applied to the data

        OUTPUTS:
        chi2: the chi2 value calculated
        '''
        # if isinstance(v, (float, int)):
        #     v = np.array([v])

        # Calculate the shifted model with the correct grid
        model_y = self.new_model(v, berv, func, i=i)

        # Remove batch dimension if present
        if model_y.dim() == 2 and model_y.size(0) == 1:
            model_y = model_y.squeeze(0)
        # Flatten to 1D
        model_y = model_y.view(-1)
        
        # Mask NaNs in the data
        valid = ~torch.isnan(data) & ~torch.isnan(model_y)

        # Ensure all arrays will have the same lengths as our valid data points       
        data_valid = data[valid]
        model_valid = model_y[valid]
        sig_valid = sig[valid]

        # Only consider the middle portions as the ends may be affected by bad interpolation
        # Will not use ends of spectrum as they will be affected by convolution 
        start = int(len(data)*0.005)
        end = int(len(data)*0.995)
        
        data_valid = data_valid[start:end]
        model_valid = model_valid[start:end]    
        sig_valid = sig_valid[start:end]
        
        # Determine the uncertainty
        sig  = sig_valid**2  #Uncertainty of observation
        if self.type == 'template':
            sig = sig*(1+1/self.Ntemp)
        # This is taken as Equation 2 from (Silva et al. 2022)
        residual = ((data_valid - model_valid))**2/sig
        chi2 = torch.sum(residual).item()
        return chi2

    def find_dv(self, data, sig, berv,func, dv=1):
        '''
        This function estimates the the planetary signal. This method is taken from (Silva et al. 2022) "A novel framework for 
        semi-Bayesian radial velocities through template matching". It is part of the S-BART methodology. 

        INPUTS:
        data: the observation spectrum
        sig: the sigma of the gaussian noise applied to the data

        OUTPUTS:
        rvorder: the proposed RV for the order
        unc_dv: the proposed dv for this iteration
        '''
        # data = data.cpu().numpy().astype(np.float64)
        # sig = sig.cpu().numpy().astype(np.float64)

        # Initialize how we will store the RVs and uncertainties
        rv_order = np.zeros_like(berv)
        unc_dv = np.zeros_like(berv)

        # Go through each observation and find the best-fit RV
        for i in range(len(berv)):
            # passing each observation index i
            result = minimize_scalar(self.chi2,args=(data[0,i],sig[0,i],berv[i],func, i),method='brent')

            # rvmin is the best-fit velocity
            # xm is the chi2 value at best fit
            rvmin, xm = result.x, result.fun
            
            if not isinstance(rvmin, float):
                rvmin= rvmin[0]
            xmp1 = self.chi2(rvmin+dv,data[0,i],sig[0,i],berv[i],func, i) #pass i
            xmm1 = self.chi2(rvmin-dv,data[0,i],sig[0,i],berv[i],func, i) #pass i
            # Equation 3 from paper
            rv_order[i] = rvmin - (dv/2)*(xmp1-xmm1)/(xmp1+xmm1-2*xm)
            unc_dv[i] = np.sqrt((2*dv**2)/(xmm1-(2*xm)+xmp1))

        return rv_order, unc_dv

    def find_unc(self,data,sig,rvmin,dv=1):
        '''
        Again this calculates the uncertainty based on equation 3 from the (Silva et al. 2022) paper.
        '''

        xm = self.chi2(rvmin,data,sig)
        xmp1 = self.chi2(rvmin+dv,data,sig)
        xmm1 = self.chi2(rvmin-dv,data,sig)
        unc_dv = np.sqrt((2*dv**2)/(xmm1-(2*xm)+xmp1))

        return unc_dv