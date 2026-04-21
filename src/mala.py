import numpy as np
from transformer import *
import torch.distributions as dist
from torch.func import grad
import torch
from torch.func import grad
from astropy.constants import c

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

c = torch.tensor(c,dtype=torch.float64,device=DEVICE)


class MALA():
    def __init__(self,obs:torch.Tensor,sig_n:torch.Tensor,berv:torch.Tensor,snr:float,inst_wgrid:torch.Tensor,spec_wgrid:torch.Tensor) -> None:
        '''
        This is to initalize the MALA object in order to do MALA sampling
        
        INTPUTS:
        obs = This is the observations used for sampling
        sig_n = This is the std of the gaussian noise estimated for the observations
        transform = This is the function used to transform the velocities

        '''
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs = obs
        self.covariance = sig_n
        self.berv = berv
        self.snr = snr
        self.inst_wgrid = inst_wgrid
        self.spec_wgrid = spec_wgrid

        # Define the start and end to only evaluate in regimes that aren't impacted by interpolation weirdness
        self.start = int(len(self.obs[0,0])*0.005)
        self.end = int(len(self.obs[0,0])*0.995)

        # Define the unpadded regions of the spectrum 
        self.non_ones = torch.where(self.spec_wgrid != 1)[0]
        # self.spec_wgrid = self.spec_wgrid[self.non_ones[0] : self.non_ones[-1] + 1]

        # Reshape the wavlenegth grids to fit the interpolating function
        # self.inst_wgrid = self.inst_wgrid #.unsqueeze(0).repeat(obs.shape[0],1).to(DEVICE)



        pass

    def find_rv(self,x_init:torch.Tensor,S:torch.Tensor,steps:int):
        '''
        This function does the sampling routine by calling the function mala_step

        INPUTS:
        x_init: current sample (this is the starting point RV which will be given by the template RVs)
        S: is the sampled spectrum
        chain: how many chains of sampling do you want
        steps: how many steps do you want to compute for the sampling 


        OUTPUTS:
        samples: the final tensor of sampled RVs
        accepted: the final tensor for which samples had accepted new steps
        '''
        # Define stepsize based on the bouchy limit uncertainty
        # Convert to specific SNR (taken from ENIRIC package) --> this makes it unitless
        A_0 = self.snr**2*self.obs[0,0].cpu().numpy()
   
        A_0 = A_0[self.start:self.end]#.numpy()

        Lambda = self.inst_wgrid.cpu().numpy()[self.start:self.end]

        # Compute the uncertainty
        dAdlam = np.gradient(A_0,Lambda)
        W = (Lambda*dAdlam)**2/A_0
        Q = np.sqrt(np.sum(W))/np.sqrt(np.sum(A_0))
        Ne = np.sum(A_0)

        self.deltaV = c/(Q*np.sqrt(Ne))
        
        target_accept_min = 0.30
        target_accept_max = 0.40
        adapt_rate = 0.2   # how aggressively to adjust step size
        adapt_window = 25  # how often to update (in steps)
        
        step_size = self.deltaV * (700 / self.snr)
        
        samples = torch.zeros((steps + 1, x_init.shape[0], x_init.shape[1]), dtype=torch.float64, device=DEVICE)
        accepted = torch.zeros((steps + 1, x_init.shape[0], x_init.shape[1]), dtype=torch.float64, device=DEVICE)
        samples[0] = x_init.clone().to(DEVICE)
        
        with torch.no_grad():
            sample = x_init.clone().to(DEVICE)
            for j in range(1, steps + 1):
                sample, accept = self.mala_step(sample, S, step_size)
                accepted[j] = accept
                samples[j] = sample
        
                # Every adapt_window steps, update step size based on acceptance
                if j % adapt_window == 0:
                    accept_rate = accepted[j - adapt_window + 1:j + 1].mean().item()
        
                    if accept_rate < target_accept_min:
                        step_size *= (1.0 - adapt_rate)
                    elif accept_rate > target_accept_max:
                        step_size *= (1.0 + adapt_rate)
        
                    # Optional: clamp to avoid numerical instability
                    step_size = max(step_size, 1e-6)
        
                    print(f"Step {j}: acceptance={accept_rate:.3f}, new step_size={step_size:.6f}")

        return samples, accepted

    def mala_step(self, x:torch.Tensor, S:torch.Tensor, step_size=1e-4, gauss=True,precond_matrix=None, rejection_step=True):
        '''
        This function proposes a new step using Langevin sampling. Langevin sampling proposes a new step by combinging the score of the probability distribution
        and a random walk, in order to walk towards regions of high probability while still including some random behaviour. Then we accept or reject this proposed step
        using the Metropolis-Hastings algorithm which is based on the actual probability of the proposed step. This improves mixing and convergence

        INPUTS:
        x: current sample
        S: is the sampled spectrum
        gauss: This is if there is just gaussian noise in the observations
        step_size: a parameter that influences how different the proposed step is
        precond_matrix: some preconditional matrix if you want to use more information than just stepsize to determine the next step
        rejection_step: Bool for if you just do langevin sampling or include Metropolis-Hastings algorithm (if True include metropolis step)

        OUTPUTS:
        x_new: the new sample
        accept: bool if a sample was rejected or not
        '''
        # If only gaussian noise, define the score function and log probability function as follows
        if gauss:
            score_fn = lambda x: self.score_gaussian(x, S, self.covariance)
            log_prob_fn =  lambda x: self.log_prob_gaussian(x, S, self.covariance)


        ## HAVE TO CONSIDER THE PROBABILITY OF THE LANGEVIN STEP? ##
        def q(xp, x, score_x):
            if precond_matrix is not None:
                precond_matrix_squared = precond_matrix @ precond_matrix.T
                mu = x + step_size * (precond_matrix_squared @ score_x)
                cov = 2 * step_size * precond_matrix_squared
                return (-0.5 * (xp - mu).T @ cov.inverse() @ (xp - mu)).sum()
            else:
                mu = x + step_size * score_x
                cov = 2 * step_size
                return (-0.5 * (xp - mu)**2/cov) #.sum()


        # Calculate Langevin step #####
        # First determine the score of the current sample
        score = score_fn(x)
        if precond_matrix is not None:
            precond_matrix_squared = precond_matrix @ precond_matrix.T
            dx = step_size * (precond_matrix_squared @ score)
            dx += np.sqrt(2*step_size) * (precond_matrix @ torch.randn_like(x))
        else:
            # The step is determined by the stepsize, score, and some randomness (langevin)
            dx = step_size * score + torch.sqrt(2*step_size) * torch.randn_like(x)
        # Proposed langevin step
        x_new = x + dx
        
        # If we only want to do Langevin step #####
        if not rejection_step:
            # Plain old Langevin
            return x_new, True

        # Metropolis-Hastings Algorithim ####
        # Compute the score including the langevin step
        score_new = score_fn(x_new)

        # Compute the ratio of probability of this proposed step vs the current step
        # This probability is based on prob of x_new in terms of your target distribution + prob of x_new in terms of langevin dynamics
        ratio = log_prob_fn(x_new) + q(x, x_new, score_new) - log_prob_fn(x) - q(x_new, x, score)

        # Accept or reject this new step based on the pre-calculated ratio
        # Create a tensor of zeros with the same size as 'ratio'
        mins = torch.zeros_like(ratio).to(DEVICE)
        log_alpha = torch.minimum(mins, ratio)

        # Compare the log random values with log_alpha
        beat = torch.rand(x.shape).log().to(DEVICE)
        condition = beat <= log_alpha
        # Use torch.where to select values based on the condition
        result = torch.where(condition, x_new,x)

        # Return the result tensor and the condition tensor
        return result, condition

    def log_prob_gaussian(self,x:torch.Tensor, S:torch.Tensor, cov:torch.Tensor
        ) -> torch.Tensor:
        # Log-probability for a Gaussian
        # You have to shift the sampled spectrum by the new sampled V 


        sampled_obs = forward_model(S,self.spec_wgrid,self.inst_wgrid,self.berv,x,change_res=True)

        # Compare this with the observations
        norm = dist.Normal(self.obs[:,:,self.start:self.end], cov[:,:,self.start:self.end])
        pdf_values = norm.log_prob(sampled_obs[:,:,self.start:self.end])
        prob_values = pdf_values.sum(axis=-1)

        return prob_values

    def log_prob_gaussian_for_score(self,x:torch.Tensor, S:torch.Tensor, cov:torch.Tensor
        ) -> torch.Tensor:
        # Log-probability for a Gaussian
        # You have to shift the sampled spectrum by the new sampled V

        sampled_obs = forward_model(S,self.spec_wgrid,self.inst_wgrid,self.berv,x,change_res=True)


        # Compare this with the observations
        norm = dist.Normal(self.obs[:,:,self.start:self.end], cov[:,:,self.start:self.end])
        pdf_values = norm.log_prob(sampled_obs[:,:,self.start:self.end])
        prob_values = pdf_values.sum()

        return prob_values

    def score_gaussian(self,x, mu, cov):
        ### CHECK TO MAKE SURE ####
        # Gradient of log-probability for a Gaussian
        scr = grad(self.log_prob_gaussian_for_score,argnums=0)(x,mu,cov)
        # scr = vmap(grad(self.log_prob_gaussian),in_dims=(None,10,10))(x,mu,cov)
        return scr