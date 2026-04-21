import numpy as np
import pandas as pd
import torch 
import pickle
from transformer import *
from convolution import *
from scipy.optimize import minimize_scalar
torch.set_default_dtype(torch.float64) 

class RV_Retrieval():
    def __init__(self, SNR,model, upsampled_wgrid, instrument_wgrid, Ntemp, type="template"):
        '''
        In this class we will find the relative velocity between the template and an observation, 
        this relative velocity should only be the planetary signal. 

        '''
        self.SNR = SNR
        self.model = model
        self.upsampled_wgrid = upsampled_wgrid
        self.instrument_wgrid = instrument_wgrid
        self.Ntemp = Ntemp
        self.type = type

        if Ntemp!=0:
            self.T_factor = 1/np.sqrt(self.Ntemp)
        else:
            self.T_factor = 0

        pass

    def new_model(self, dv, berv,func):
        '''
        This function is used to shift the template by various dv values that represent 
        different planetary signals. This shifted model will then be compared to the true observation
        signal to estimate the likilihood of the RV value

        INPUTS:
        dv: an array of radial velocities that encompass all values that planetary signal might be 

        OUTPUTS:
        shifted_model: an array of the template shifted by the various dv values
        '''
        # if not isinstance(dv, float):
        #     dv = dv[0]
 
        rv = dv+berv
        if rv.ndim==0:
            rv = torch.tensor([rv]).to(DEVICE)
        if self.type=='template':

            shifted = shift_spectrum(self.model.view(1, 1, -1),rv.unsqueeze(0),self.upsampled_wgrid.unsqueeze(0).unsqueeze(0),func)
            batched_wgrid = self.upsampled_wgrid.unsqueeze(0).unsqueeze(0)#.repeat(self.broadened_observations.shape[0],1).to(DEVICE)
            batched_instwgrid = self.instrument_wgrid.unsqueeze(0).unsqueeze(0)
            shifted_degraded_template = interpolate(batched_wgrid,shifted,batched_instwgrid,func)

        elif self.type=='intrinsic':
            envelope = torch.quantile(self.model,q=0.5).to(DEVICE)
            right_flux = self.model.clone().detach()/envelope
            shifted = shift_spectrum(right_flux.view(1, 1, -1),rv.unsqueeze(0),self.upsampled_wgrid.unsqueeze(0).unsqueeze(0),func)
            broadened = torch.tensor(gauss_convolve(self.upsampled_wgrid.cpu().numpy().astype(np.float64),shifted[0,0].cpu().numpy().astype(np.float64),70_000, 
                                                    n_fwhm=7, res_rtol=1e-6, mode='same', i_plot=True)[1],dtype=torch.float64).to(DEVICE)
            
            batched_wgrid = self.upsampled_wgrid.unsqueeze(0).unsqueeze(0)#.repeat(self.broadened_observations.shape[0],1).to(DEVICE)
            batched_instwgrid = self.instrument_wgrid.unsqueeze(0).unsqueeze(0)
            shifted_degraded_template = interpolate(batched_wgrid,broadened.unsqueeze(0).unsqueeze(0),batched_instwgrid,func)

        
        elif self.type=="sample":
            right_flux = self.model
            shifted = shift_spectrum(right_flux.view(1, 1, -1),rv.unsqueeze(0),self.upsampled_wgrid.unsqueeze(0).unsqueeze(0),func)
            batched_wgrid = self.upsampled_wgrid.unsqueeze(0).unsqueeze(0)#.repeat(self.broadened_observations.shape[0],1).to(DEVICE)
            batched_instwgrid = self.instrument_wgrid.unsqueeze(0).unsqueeze(0)
            shifted_degraded_template = interpolate(batched_wgrid,shifted,batched_instwgrid,func)


        return shifted_degraded_template.cpu()

    def chi2(self,v,data,sig,berv,func):
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
        # Calculate the shifted model
        model_y = self.new_model(v,berv,func)[0]

        # Only consider the middle portions as the ends may be affected by bad interpolation
        start = int(len(data)*0.1)
        end = int(len(data)*0.9)


        # Determine the uncertainty
        sigt = (sig*self.T_factor)**2 # Uncertainty of template
        sigo  = sig**2 # Uncertainty of observation
        sig = sigo+sigt
        # Will not use ends of spectrum as they will be affected by convolution 
        # This is taken as Equation 2 from (Silva et al. 2022)
        residual = ((data[start:end]-model_y[:,start:end]))**2/sig[start:end]
        chi2 = torch.sum(residual,axis=1)
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


        rv_order = np.zeros_like(berv)
        unc_dv = np.zeros_like(berv)
        for i in range(len(berv)):

            result = minimize_scalar(self.chi2,args=(data[0,i],sig[0,i],berv[i],func),method='brent')
            rvmin, xm = result.x, result.fun
            
            if not isinstance(rvmin, float):
                rvmin= rvmin[0]
            xmp1 = self.chi2(rvmin+dv,data[0,i],sig[0,i],berv[i],func)
            xmm1 = self.chi2(rvmin-dv,data[0,i],sig[0,i],berv[i],func)

            # Equation 3 from paper
            rv_order[i] = rvmin - (dv/2)*(xmp1-xmm1)/(xmp1+xmm1-2*xm)
            unc_dv[i] = (2*dv**2)/(xmm1-(2*xm)+xmp1) 

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