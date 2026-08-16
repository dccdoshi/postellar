import numpy as np
from transformer import *
import torch.distributions as dist
from torch.func import grad
import torch
from astropy.constants import c

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

c = torch.tensor(c, dtype=torch.float64, device=DEVICE)

class MALA():
    def __init__(self, obs: torch.Tensor, sig_n: torch.Tensor, berv: torch.Tensor, snr, inst_wgrid: torch.Tensor,
                 spec_wgrid: torch.Tensor, sys_vel: torch.Tensor, mask: torch.Tensor = None) -> None:
        '''
        This is to initalize the MALA object in order to do MALA sampling

        INPUTS:
        obs = This is the observations used for sampling
        sig_n = This is the std of the gaussian noise estimated for the observations. In real data this is 1/SNR
        berv = the bervs of your observation
        snr = the snr of your observations
        REAL DATA UPDATE - snr can now be a single float OR a per-observation tensor. The step size is computed from this, and needs each obs appropriate SNR
        inst_wgrid = instruments/observation's wavelength grids
        spec_wgrid = wavelength grid of spectrum parameter (PHOENIX grid)
        REAL DATA UPDATE - mask: updated to take in a mask to perform the MALA sampling over only the valid pixels
        REAL DATA UPDATE - sys_vel: explicitly passing the systemic velocity is also important now, used in the forward model
        '''
        self.obs = obs
        self.covariance = sig_n
        self.berv = berv
        self.inst_wgrid = inst_wgrid
        self.spec_wgrid = spec_wgrid
        self.sys_vel = sys_vel

        # REAL DATA UPDATE - snr can be accepted as either a scalar or a per-observation tensor
        if torch.is_tensor(snr):
            self.snr = snr.to(DEVICE).flatten().double()
        else:
            self.snr = torch.as_tensor(snr, dtype=torch.float64, device=DEVICE).flatten()

        # Fallback case for synthetic data
        # Define the start and end to only evaluate in regimes that aren't impacted by interpolation weirdness
        self.start = int(len(self.obs[0, 0]) * 0.005)
        self.end = int(len(self.obs[0, 0]) * 0.995)

        if mask is not None:
            self.mask = mask.bool().to(DEVICE)
        else:
            self.mask = None

        # Define the unpadded regions of the spectrum
        self.non_ones = torch.where(self.spec_wgrid != 1)[0]
        pass

    def find_rv(self, x_init: torch.Tensor, S: torch.Tensor, steps: int):
        '''
        This function does the sampling routine by calling the function mala_step
        
        INPUTS:
        x_init: current sample (this is the starting point RV which will be given by the template RVs)
        S: is the sampled spectrum
        steps: how many steps do you want to compute for the sampling 


        OUTPUTS:
        samples: the final tensor of sampled RVs
        accepted: the final tensor for which samples had accepted new steps

        REAL DATA UPDATE - previously the step size was computed ONCE, using only self.obs[0,0] and a shared snr
        It was then applied identically to every observation.
        This is no longer valid as real data will have varying SNRs
        Now looped per observation below, each using its own snr and its own data. step_size is shape [1, N].
        '''
        N = self.obs.shape[1]  #number of observations

        if self.inst_wgrid.dim() == 1:  # Case for synthetic data
            inst_wgrid_per_obs = self.inst_wgrid.unsqueeze(0).expand(N, -1)
        else:                           # Real data
            inst_wgrid_per_obs = self.inst_wgrid

        if self.snr.numel() == 1:  #Synthetic data has one consistent SNR
            snr_arr = self.snr.expand(N)
        else:                      # Real data has a varying SNR
            snr_arr = self.snr

        if self.mask is not None:
            mask_np_common = self.mask[0, 0].cpu().numpy()
        else:
            mask_np_common = None

        # REAL DATA UPDATE - step size needs to be computed PER OBSERVATION, not once globally
        deltaVs = torch.zeros(N, dtype=torch.float64, device=DEVICE)
        step_sizes = torch.zeros(N, dtype=torch.float64, device=DEVICE)

        # Calculate bouchy uncertainty
        # Convert to specific SNR (taken from ENIRIC package) --> this makes it unitless
        for i in range(N):
            snr_i = float(snr_arr[i].item())
            Lambda_full = inst_wgrid_per_obs[i].cpu().numpy()
            obs_i_np = self.obs[0, i].cpu().numpy()
            A_full = snr_i ** 2 * obs_i_np

            if mask_np_common is not None:   # for real data
                base_mask_i = mask_np_common
            else:                            # for synthetic data using the start/end parameters
                base_mask_i = np.zeros(len(obs_i_np), dtype=bool)
                base_mask_i[self.start:self.end] = True

            # REAL DATA UPDATE - interior NaN check on this observation's own data
            finite_i = np.isfinite(obs_i_np)
            # Combines the two masks, the common range and the interior NaN checks. Any NaNs in the gradients causes it to break
            combined_mask_i = base_mask_i & finite_i

            # Subset to only valid data
            A_0 = A_full[combined_mask_i]
            Lambda = Lambda_full[combined_mask_i]

            # Compute the uncertainty
            dAdlam = np.gradient(A_0, Lambda)
            W = (Lambda * dAdlam) ** 2 / A_0
            Q = np.sqrt(np.sum(W)) / np.sqrt(np.sum(A_0))
            Ne = np.sum(A_0)

            deltaV_i = (c / (Q * np.sqrt(Ne))).item()
            deltaVs[i] = deltaV_i
            step_sizes[i] = deltaV_i * (700.0 / snr_i)

        self.deltaV_per_obs = deltaVs
        self.deltaV = deltaVs[0]  #observation 0's only

        step_size = step_sizes.clamp(min=1e-6).view(1, N)

        # Parameters to implement adaptive stepsize
        target_accept_min = 0.30
        target_accept_max = 0.40
        adapt_rate = 0.2  # how aggressively to adjust step size
        adapt_window = 25 # how often to update (in steps)

        samples = torch.zeros((steps + 1, x_init.shape[0], x_init.shape[1]), dtype=torch.float64, device=DEVICE)
        accepted = torch.zeros((steps + 1, x_init.shape[0], x_init.shape[1]), dtype=torch.float64, device=DEVICE)
        samples[0] = x_init.clone().to(DEVICE)

        with torch.no_grad():
            sample = x_init.clone().to(DEVICE)
            for j in range(1, steps + 1):
                sample, accept = self.mala_step(sample, S, step_size)
                accepted[j] = accept
                samples[j] = sample

                # REAL DATA UPDATE - this update is now PER OBSERVATION (torch.where across [1,N]),
                # rather than a single shared scalar accept_rate/step_size update
                if j % adapt_window == 0:
                    accept_rate = accepted[j - adapt_window + 1: j + 1].mean(dim=(0, 1)).view(1, N)
                    # REAL DATA UPDATE - torch.where instead of the if/elif statements. This is since accept_rate is [1,N] and each obs is adjusted independently 
                    step_size = torch.where(accept_rate < target_accept_min, step_size * (1.0 - adapt_rate), step_size)
                    step_size = torch.where(accept_rate > target_accept_max, step_size * (1.0 + adapt_rate), step_size)

                    # Optional: clamp to avoid numerical instability
                    # REAL DATA UPDATE - step_size is now a tensor, [1, N] so clamp instead of max()
                    step_size = step_size.clamp(min=1e-6)
                    print(f"Step {j}: mean acceptance={accept_rate.mean().item():.3f}, "
                          f"step_size range=[{step_size.min().item():.6f}, {step_size.max().item():.6f}]")

        return samples, accepted

    def mala_step(self, x: torch.Tensor, S: torch.Tensor, step_size=1e-4, gauss = True, precond_matrix = None, rejection_step=True):    
        '''
        This function proposes a new step using Langevin sampling. Langevin sampling proposes a new step by combining the score of the probability distribution
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
            log_prob_fn = lambda x: self.log_prob_gaussian(x, S, self.covariance)

        ## HAVE TO CONSIDER THE PROBABILITY OF THE LANGEVIN STEP to maintain detailed balance ##
        def q(xp, x, score_x):
            if precond_matrix is not None:
                precond_matrix_squared = precond_matrix @ precond_matrix.T
                mu = x + step_size * (precond_matrix_squared @ score_x)
                cov = 2 * step_size * precond_matrix_squared
                return (-0.5 * (xp - mu).T @ cov.inverse() @ (xp - mu)).sum()
            else:
                mu = x + step_size * score_x
                cov = 2 * step_size
                return (-0.5 * (xp - mu) ** 2 / cov)
            
        # Calculate Langevin step #####
        # First determine the score of the current sample
        score = score_fn(x)
        if precond_matrix is not None:
            precond_matrix_squared = precond_matrix @ precond_matrix.T
            dx = step_size * (precond_matrix_squared @ score)
            dx += np.sqrt(2 * step_size) * (precond_matrix @ torch.randn_like(x))
        else:
            # The step is determined by the stepsize, score, and some randomness (langevin)
            dx = step_size * score + torch.sqrt(2 * step_size) * torch.randn_like(x)

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
        # This probability is based on prob of x_new in terms of your target distribution
        # + prob of x_new in terms of langevin dynamics
        ratio = log_prob_fn(x_new) + q(x, x_new, score_new) - log_prob_fn(x) - q(x_new, x, score)
        log_alpha = torch.minimum(torch.zeros_like(ratio), ratio)

        # Compare the log random values with log_alpha
        beat = torch.rand(x.shape).log().to(DEVICE)
        condition = beat <= log_alpha

        # Use torch.where to select values based on the condition.
        # REAL DATA UPDATE - condition is [B, N], so each draw and each observation is
        # accepted or rejected independently. This depends on log_prob_gaussian returning
        # [B, N] rather than summing to a scalar.
        result = torch.where(condition, x_new, x)

        # Return the result tensor and the condition tensor
        return result, condition

    def log_prob_gaussian(self, x: torch.Tensor, S: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        '''
        Log-probability of the observations under the model, for a proposed set
        of velocities x. The sampled spectrum S is shifted by x through the
        forward model and compared against the data.

        INPUTS:
        x: the proposed velocities, [B, N]
        S: the sampled spectrum
        cov: the per-pixel uncertainty

        OUTPUTS:
        [B, N] -- one log-probability per draw per observation

        REAL DATA UPDATE - The [B, N] shape is the important part! sum(dim=-1) sums over pixels only, leaving each draw and observation separate, 
        so mala_step accepts or rejects each one independently

        REAL DATA UPDATE - the mask branch below. Real observations have NaNs and dist.Normal propagates them, 
        so the bad pixels are replaced with safe values before the log_prob and zeroed out afterwards rather than
        being indexed out -- which keeps the shape fixed for torch.func.grad.
        '''
        # REAL DATA UPDATE - Updated to take in sys_vel
        sampled_obs = forward_model(S, self.spec_wgrid, self.inst_wgrid, self.berv, x, sys_vel=self.sys_vel)

        if self.mask is not None:  # for real data
            mask_use = self.mask
            # The mask is [1, 1, L]. Guard in case a larger leading dimension
            # ever arrives -- only the first is wanted.
            if mask_use.shape[0] != 1:
                mask_use = mask_use[0:1]

            obs_slice = self.obs
            model_slice = sampled_obs
            cov_slice = cov

            # There is one model per draw but only one set of observations, so obs and cov are [1, N, L] while the model is [B, N, L]
            if model_slice.shape[0] != obs_slice.shape[0]:
                obs_slice = obs_slice.expand(model_slice.shape[0], -1, -1)
                cov_slice = cov_slice.expand(model_slice.shape[0], -1, -1)
            if mask_use.shape[0] != model_slice.shape[0]:
                mask_use = mask_use.expand(model_slice.shape[0], -1, -1)

            # Four conditions: the common range from the mask, plus interior NaNs in the data, NaN the forward model produced at its own edges, and
            # any bad uncertainties
            finite_mask = mask_use & torch.isfinite(obs_slice) & torch.isfinite(model_slice) & torch.isfinite(cov_slice)


            # REAL DATA UPDATE - the bad pixels get placeholder values rather than being removed. 
            # This was done because indexing gives a flat, variable-length array and loses the [B, N, L]
            # shape that sum(dim=-1) and torch.func.grad both need.
            # obs and model get 0.0 -- the value does not matter, since these pixels are zeroed out of the sum
            # cov gets 1.0 and NOT 0.0, because dist.Normal divides by the scale and a zero there gives infinity.

            safe_obs = torch.where(finite_mask, obs_slice, torch.zeros_like(obs_slice))
            safe_model = torch.where(finite_mask, model_slice, torch.zeros_like(model_slice))
            safe_cov = torch.where(finite_mask, cov_slice, torch.ones_like(cov_slice))

            norm = dist.Normal(safe_obs, safe_cov)
            pdf_values = norm.log_prob(safe_model)
            pdf_values = torch.where(finite_mask, pdf_values, torch.zeros_like(pdf_values))
            return pdf_values.sum(dim=-1)
        
        else:  #synthetic data, no NaN handling required
            obs_slice = self.obs[:, :, self.start:self.end]
            model_slice = sampled_obs[:, :, self.start:self.end]
            cov_slice = cov[:, :, self.start:self.end]
            norm = dist.Normal(obs_slice, cov_slice)
            pdf_values = norm.log_prob(model_slice)
            return pdf_values.sum(dim=-1)

    def log_prob_gaussian_for_score(self, x: torch.Tensor, S: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        '''
        REAL DATA UPDATE - kept SEPARATE from log_prob_gaussian on purpose: torch.func.grad
        requires a SCALAR-valued function, but log_prob_gaussian above returns [B,N]. This sums
        over EVERYTHING into one number, used only for the gradient in score_gaussian below.
        '''
        sampled_obs = forward_model(S, self.spec_wgrid, self.inst_wgrid, self.berv, x, sys_vel=self.sys_vel)

        if self.mask is not None:  # real data
            # See log_prob_gaussian for what all of this does - it is the same 
            mask_use = self.mask
            if mask_use.shape[0] != 1:
                mask_use = mask_use[0:1]

            obs_slice = self.obs
            model_slice = sampled_obs
            cov_slice = cov
            if model_slice.shape[0] != obs_slice.shape[0]:
                obs_slice = obs_slice.expand(model_slice.shape[0], -1, -1)
                cov_slice = cov_slice.expand(model_slice.shape[0], -1, -1)
            if mask_use.shape[0] != model_slice.shape[0]:
                mask_use = mask_use.expand(model_slice.shape[0], -1, -1)

            finite_mask = mask_use & torch.isfinite(obs_slice) & torch.isfinite(model_slice) & torch.isfinite(cov_slice)

            # Placeholders for the bad pixels
            safe_obs = torch.where(finite_mask, obs_slice, torch.zeros_like(obs_slice))
            safe_model = torch.where(finite_mask, model_slice, torch.zeros_like(model_slice))
            safe_cov = torch.where(finite_mask, cov_slice, torch.ones_like(cov_slice))

            norm = dist.Normal(safe_obs, safe_cov)
            pdf_values = norm.log_prob(safe_model)
            pdf_values = torch.where(finite_mask, pdf_values, torch.zeros_like(pdf_values))

            # sum() over everything, unlike sum(dim=-1) in log_prob_gaussian --
            # torch.func.grad needs a scalar
            return pdf_values.sum()
        else:  #synthetic data
            obs_slice = self.obs[:, :, self.start:self.end]
            model_slice = sampled_obs[:, :, self.start:self.end]
            cov_slice = cov[:, :, self.start:self.end]
            norm = dist.Normal(obs_slice, cov_slice)
            pdf_values = norm.log_prob(model_slice)
            return pdf_values.sum()

    def score_gaussian(self, x, mu, cov):
        # Gradient of log-probability for a Gaussian
        scr = grad(self.log_prob_gaussian_for_score, argnums=0)(x, mu, cov)
        return scr