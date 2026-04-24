import numpy as np
import pandas as pd
import torch 
import pickle
from transformer import *
from convolution import *
import PyAstronomy.pyasl as pya 
from astropy.time import Time 
from astropy.io import fits
from astropy import constants as const

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64) 

class Observations():
    '''
    Here is the code used to make synthetic observations from our validation dataset
    '''

    def __init__(self, i: int = 5, N: int = 10, kamp:float = 0, SNR = 100,seed:int = 5,inst_res:int = 70_000,order=20,
                filepath: str = '../data/SPIRou20_val.df',wfile="../data/SPIRou_wavelength_solution.fits",star='proxima') -> None:
        '''
        This defines the true stellar spectrum that will be used to create your synthetic observations. Our test data
        is the dataframe saved from a py file in create spectra. The shape is D where D is the length of the spectrum.
        Wgrid has a padded length, we will remove the padding to make sure it is length D. 

        INPUTS:
        i = indicate which sample you want to use from your test data
        N = number of observations we want to make
        kamp = planetary k amplitude
        SNR = snr of observations
        seed = seed for noise 
        inst_res = broadening factor
        filepath = filepath to your test data

        OUTPUTS:
        obs = the observations of shape [N,L]
        '''
        self.N = N
        self.kamp = kamp
        self.SNR = SNR
        self.seed = seed
        self.inst_res = inst_res
        self.order = order

        ## Retrieve native wavelength grid 
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.wgrid = torch.tensor(data['Wavelength'].iloc[0],dtype=torch.float64).to(DEVICE)
        non_ones = torch.where(self.wgrid != 1)[0]
        self.padded_wgrid = self.wgrid.clone().detach().to(DEVICE)
        self.wgrid = self.wgrid[non_ones[0] : non_ones[-1] + 1]


        ## Upload empirical templates of the stars and save their respective systemic velocities (we will have to get rid of this overall shift)
        if star=='barnards':
            filename = "../data/Template_s1dv_GL699_sc1d_v_file_AB.fits"
            RV = torch.tensor(-110.47*1e3).unsqueeze(0)
        else:
            filename = "../data/Template_s1dv_PROXIMA_sc1d_v_file_A.fits"
            RV = torch.tensor(-22204).unsqueeze(0)
        hdu  = fits.open(filename)
        
        #Headers from Primary HDU (No Data only Information)
        headers = hdu[0].header 
        
        #Contains the data from file
        # Extract the data as a NumPy array
        data_array = hdu[1].data
        
        # Convert all fields in the structured array to native byte order
        data_array = data_array.byteswap().newbyteorder()
        
        # Create DataFrame
        nandata = pd.DataFrame(np.array(data_array))
        
        # Drop NaNs safely
        self.star_flux = nandata["flux"].dropna()
        self.star_wavelength = nandata.loc[self.star_flux.index, "wavelength"]
        # --- Convert numpy ➜ torch properly (no warnings)
        star_flux_torch = torch.from_numpy(self.star_flux.to_numpy()).float().unsqueeze(0).unsqueeze(0)
        star_wave_torch = torch.from_numpy(self.star_wavelength.to_numpy()).float().unsqueeze(0).unsqueeze(0)
        wgrid_torch     = self.wgrid.unsqueeze(0).unsqueeze(0)

        # 1. Shift spectrum
        speed_of_light_ms = const.c.value
        # relativistic calculation (1 - v/c)
        part1 = 1 - (RV / speed_of_light_ms)
        # relativistic calculation (1 + v/c)
        part2 = 1 + (RV / speed_of_light_ms)

        shifted_grid = star_wave_torch * torch.sqrt(part1 / part2)

        # 2. Interpolate onto wgrid
        self.star_final = interpolate(
            shifted_grid,
            star_flux_torch,
            wgrid_torch,
            func='scipy'
        )

        # 3. Normalize by median
        wavelength = self.wgrid.cpu().numpy()
        flux = self.star_final[0].cpu().numpy()
        nbins = 100  # adjust depending on how dense your spectrum is
        bins = np.linspace(wavelength.min(), wavelength.max(), nbins + 1)
        inds = np.digitize(wavelength, bins)
        
        w_med = np.array([np.median(wavelength[inds == i]) for i in range(1, nbins+1) if np.any(inds == i)])
        f_med = np.array([np.median(flux[inds == i]) for i in range(1, nbins+1) if np.any(inds == i)])
        
        # Step 2: fit a linear function to the median points
        coeffs = np.polyfit(w_med, f_med, 1)
        fit = np.poly1d(coeffs)

        # Step 3: normalize
        flux_norm = self.star_final / torch.tensor(fit(wavelength),device=DEVICE)
        flux_norm = flux_norm / torch.quantile(flux_norm,0.5)


        # 4. Store original spectrum — no torch.tensor()!
        self.original_spectrum = (
            flux_norm.clone().detach()
            .double()
            .to(DEVICE)
            .view(1, 1, -1)
        )

        # Get instrument wavelength grid
        e = fits.open(wfile)
        wgrid =  np.sort(e[1].data[self.order]) 
        self.inst_wgrid = torch.tensor(np.ascontiguousarray(wgrid.byteswap().newbyteorder()),dtype=torch.float64).to(DEVICE)

        # Get the min and max bounds from a
        lower = self.wgrid[0]
        upper = self.wgrid[-1]

        # Select only the part of b within [lower, upper]
        self.inst_wgrid = self.inst_wgrid[(self.inst_wgrid >= lower) & (self.inst_wgrid <= upper)]

        data = None 
        non_ones = None
        pass
        

    def make_observations(self,func, add_RV= True):
        '''
        This function makes the synthetic observations using a template spectrum. 
        It will shift the observations according to an RV curve, degrade to instrument sampling resolution, 
        and then add photon noise. 

        INPUTS:
        func = the interpolation function used when creating the observations

        OUTPUTS:
        observations: observations of size [N,L] where N is number of observations and L is length of spectrum (unpadded)
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
        self.right_flux = self.SNR**2*(self.original_spectrum.clone().detach())

        # Shift the spectrum 
        RV = RV.unsqueeze(0)
        spec_wgrid_batched = self.wgrid.view(1, 1, len(self.wgrid)).expand(1, self.N,len(self.wgrid))
        self.shifted_observations = shift_spectrum(self.right_flux , RV,spec_wgrid_batched,func)

        # Theres no additional broadening we include since the templates already include the instrumental broadening

        # Interpolate to instrument sampling resolution
        spec_wgrid_batched = self.wgrid.view(1, 1, len(self.wgrid)).expand(1, len(RV), len(self.wgrid)).to(DEVICE)
        inst_wgrid_batched = self.inst_wgrid.view(1, 1, len(self.inst_wgrid)).expand(1, len(RV), len(self.inst_wgrid)).to(DEVICE)
        self.degraded_observations = interpolate(spec_wgrid_batched,self.shifted_observations,inst_wgrid_batched,func)

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

    

     
