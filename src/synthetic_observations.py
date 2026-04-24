import numpy as np
import pandas as pd
import torch 
import pickle
from transformer import *
from convolution import *
import PyAstronomy.pyasl as pya 
from astropy.time import Time 
from astropy.io import fits
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64) 

class Observations():
    '''
    Here is the code used to make synthetic observations from our validation dataset
    '''

    def __init__(self, i: int = 5, N: int = 10, kamp:float = 0, SNR = 100,seed:int = 5,inst_res:int = 70_000,order=20,
                filepath: str = '../data/SPIRou20_val.df',wfile="../data/SPIRou_wavelength_solution.fits") -> None:
        '''
        This defines the true stellar spectrum that will be used to create your synthetic observations. Our test data
        is the dataframe saved from a py file in create spectra. The shape is [D] where D is the length of the spectrum.
        Wgrid has a padded length, we will remove the padding to make sure it is length D. 

        INPUTS:
        i = indicate which sample you want to use from your test data
        N = number of observations we want to make
        kamp = planetary k amplitude
        SNR = snr of observations
        seed = seed for noise 
        inst_res = broadening factor
        order = order that you are using
        filepath = filepath to your validation data
        wfile = the wavelength solution file
        '''
        self.N = N
        self.kamp = kamp
        self.SNR = SNR
        self.seed = seed
        self.inst_res = inst_res
        self.order = order

        ## Retrieve spectrum
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # This is the spectrum straight from PHOENIX now in shape [1,1,L]
        self.original_spectrum = torch.tensor(data['Spectrum'].iloc[i],dtype=torch.float64).to(DEVICE).view(1, 1, -1)



        # This is the wgrid but it is padded at the ends, the final wgrid we want should be without the padding
        # In shape [L]
        self.wgrid = torch.tensor(data['Wavelength'].iloc[i],dtype=torch.float64).to(DEVICE)
        non_ones = torch.where(self.wgrid != 1)[0]
        self.padded_wgrid = self.wgrid.clone().detach().to(DEVICE)
        self.wgrid = self.wgrid[non_ones[0] : non_ones[-1] + 1]




        # Get the wavelength grid for the observations
        e = fits.open(wfile)
        wgrid =  np.sort(e[1].data[self.order]) 
        self.inst_wgrid = torch.tensor(np.ascontiguousarray(wgrid.byteswap().newbyteorder()),dtype=torch.float64).to(DEVICE)

        # Just making sure that our observation wgrid matches our intrinsic wgrid
        # Get the min and max bounds from a
        lower = self.wgrid[0]
        upper = self.wgrid[-1]

        # Select only the part of b within [lower, upper]
        self.inst_wgrid = self.inst_wgrid[(self.inst_wgrid >= lower) & (self.inst_wgrid <= upper)]



        # This is what the training sample would have looked like (includes instrumental broadening)
        self.training = torch.tensor(data['Final'].iloc[i],dtype=torch.float64).to(DEVICE).view(1, 1, -1)#[non_ones[0] : non_ones[-1] + 1].view(1, 1, -1)

        # This is the normalization factor applied to create the training sample
        self.normalization_factor = torch.tensor(data['Normalization_Factor'].iloc[i],dtype=torch.float64).to(DEVICE)

        data = None 
        non_ones = None
        pass
        

    def make_observations(self,func, add_RV= True):
        '''
        This function makes the synthetic observations using a normalized PHOENIX spectrum. 
        It will shift the observations according to an RV curve, broaden it, degrade to instrument sampling resolution, 
        and then add photon noise. 

        INPUTS:
        func = the interpolation function used when creating the observations

        OUTPUTS:
        observations: observations of size [N,L] where N is number of observations and L is length of 
        '''

        dates = self.define_dates()
        
        RV, planet = self.make_RV_signal(self.kamp)
        if not add_RV:
            self.berv = 0
            self.RV = 0
            self.planet = 0
            RV = RV* 0 
            planet = planet * 0

        # Flux normalization
        self.right_flux = self.SNR**2*(self.original_spectrum.clone().detach()/self.normalization_factor )

        # Shift the spectrum with injected RV signal
        RV = RV.unsqueeze(0)
        spec_wgrid_batched = self.wgrid.view(1, 1, len(self.wgrid)).expand(1, self.N,len(self.wgrid))
        self.shifted_observations = shift_spectrum(self.right_flux , RV,spec_wgrid_batched,func)

        # Broaden the normalized spectrum 
        # This code is written in numpy so send tensors to numpy and then bring back to torch 
        self.broadened_observations = torch.zeros(self.shifted_observations.shape,dtype=torch.float64).to(DEVICE)
        for i in range(len(self.shifted_observations[0])):
            self.broadened_observations[0,i] = torch.tensor(gauss_convolve(self.wgrid.cpu().numpy().astype(np.float64),self.shifted_observations[0,i].cpu().numpy().astype(np.float64),self.inst_res, 
                                                n_fwhm=7, res_rtol=1e-6, mode='same', i_plot=True)[1],dtype=torch.float64).to(DEVICE)


        # Place into sampling resolution of instrument
        spec_wgrid_batched = self.wgrid.view(1, 1, len(self.wgrid)).expand(1, len(RV), len(self.wgrid)).to(DEVICE)
        inst_wgrid_batched = self.inst_wgrid.view(1, 1, len(self.inst_wgrid)).expand(1, len(RV), len(self.inst_wgrid)).to(DEVICE)
        self.degraded_observations = interpolate(spec_wgrid_batched,self.broadened_observations,inst_wgrid_batched,func)

        # Place in SNR and add photon noise
        self.noisy_observations = self.degraded_observations.clone().detach() 
        sig = torch.sqrt(torch.abs(self.noisy_observations))
        if self.seed is not None:
            torch.manual_seed(self.seed)
        noise = sig*torch.normal(0,1,size=self.noisy_observations.size(),dtype=torch.float64).to(DEVICE)
        self.noise = noise
        self.noisy_observations += noise

        self.final_observations = self.noisy_observations
        self.uncertainty = torch.sqrt(torch.abs(self.noisy_observations))
        
        return self.final_observations, self.uncertainty

    def post_process(self):
        '''
        We need normalized data to put through the spectrum sampler. Here we will just divide our observations by what their median
        flux values should be (S/N ^2). The uncertainties should also be propogated accordingly.
        '''
        self.normalized_observations = self.final_observations/(self.SNR**2)
        self.normalized_uncertainty = self.uncertainty / (self.SNR**2)
        return self.normalized_observations, self.normalized_uncertainty

    def define_dates(self,start_date='2020-01-01',end_date='2021-01-01'):
        '''
        This function creates a julian date grid based on a given start date and end date in datetime format
        It returns the grid in datetime format however the julian date grid is used for BERV calculations. 
        The dates are slightly randomized to best replicate real observations.

        INPUTS:
        start_date: when the observations start in datetime format
        end_date: when the observations end in datetime format

        OUTPUTS:
        self.julian_dates: the grid space of dates that will be used for observations
        '''
        # Create Dates 
        # Convert start and end dates to Julian dates
        start_jd = Time(start_date).jd
        end_jd = Time(end_date).jd

        # Create ordered julian dates
        self.julian_dates = torch.linspace(start_jd, end_jd, self.N).to(DEVICE)

        return self.julian_dates

    def make_RV_signal(self,planet_amp,syst_velo=0,observational_params=[-70.73, -29.26, 217.37, -62.67, 2400.0]):
        '''
        Here we create an RV signal for the time grid that we are using
        This can include berv, the system velocity, and the planet amplitude

        INPUTS:
        syst_velo: (should be in km/s) and is the system's relative velocity
        planet amp: (should be in m/s) are floats and is the amplitude of the planet's signal
        observational_params: coordinates to dictate BERV measurements

        OUTPUTS: 
        self.RV: This is the RV signal for each of the specified dates provided
        '''
        self.long = observational_params[0]
        self.lat = observational_params[1]

        self.ra = observational_params[2]
        self.dec = observational_params[3]
        self.alt = observational_params[4]

        # Initialize Signal
        self.RV = torch.zeros(len(self.julian_dates)).to(DEVICE)

        # Add Berv
        berv = torch.zeros(len(self.julian_dates)).to(DEVICE)
        for i in range(len(self.julian_dates)):
            b = pya.helcorr(self.long,self.lat,self.alt,self.ra,self.dec,float(self.julian_dates[i]))
            berv[i] = b[0]*1e3 #m
        self.berv = berv


        # Add System Velocity
        syst_velo = syst_velo * 1e3 #m
        self.sys = syst_velo

        # Add Planet (just a simple sin curve)
        A = planet_amp
        w=0.08
        self.planet = (A*torch.sin(w*self.julian_dates)).to(DEVICE)

        # Add it all together
        self.RV=self.planet+self.sys+self.berv

        return self.RV, self.planet

    

     
