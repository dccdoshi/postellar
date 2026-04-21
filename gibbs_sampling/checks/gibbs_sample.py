import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
import torch
import matplotlib.pylab as plt
from synthetic_observations import Observations
from gaussian_synthetic_observations import Gaussian_Observations
from transformer import *
from spectrum_lsf import Score_Likelihood
from score_models import ScoreModel
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from template import Template
from sbart_rv_finder import RV_Retrieval
from mala import MALA
import h5py
from tqdm import tqdm
from torch.autograd.functional import jacobian
import argparse

# -------------------
# Parse input arguments
# -------------------
parser = argparse.ArgumentParser(description="Do Posterior Joint Sampling of RVs and Spectrum for multiple seeds")
parser.add_argument('--val_data', type=str, default='SPIRou20_val.df', help='Training data file')
parser.add_argument('--B', type=int, default=16, help='Number of Chains')
parser.add_argument('--snr', type=int, default=50, help="SNR")
parser.add_argument('--N', type=int, default=5, help="Number of Observations")
parser.add_argument('--gibbs_steps', type=int, default=200, help='Number of gibbs steps')
parser.add_argument('--checkpoints_directory', type=str, default='gausb64nf64ch2_spir4096moredownsample', help='Directory to load checkpoints')
parser.add_argument('--output', type=str, default='output.h5', help='HDF5 file to save data')
parser.add_argument('--num_seeds', type=int, default=3, help='Number of different seeds to run')
args = parser.parse_args()

# -------------------
# Load model once
# -------------------
model = ScoreModel(checkpoints_directory="../trained_models/"+args.checkpoints_directory, device=device)
change_res = True

# -------------------
# Open single HDF5 file for all seeds
# -------------------
with h5py.File("results/"+args.output, "w") as f_out:
    for seed in range(args.num_seeds):
        print(f"\n===== Running seed {seed} =====",flush=True)

        # Create observations with seed
        obs = Observations(i=0,seed=seed, SNR=args.snr, filepath="../data/"+args.val_data, N=args.N)
        synthetic_spectra, uncertainty = obs.make_observations(func='connors', add_RV=True, change_res=change_res)
        synthetic_spectra, uncertainty = obs.post_process()
        non_ones = torch.where(obs.padded_wgrid!=1)

        # Create template
        temp = Template(synthetic_spectra, obs.berv, obs.inst_wgrid, obs.wgrid)
        template = temp.make_template(func='scipy')

        # Find template RVs
        sbart = RV_Retrieval(args.snr, template, obs.wgrid, obs.inst_wgrid, args.N)
        templatervs, uncs = sbart.find_dv(synthetic_spectra[:,:].cpu(), uncertainty.cpu()[:,:], obs.berv[:].cpu(), func='connors')

        # Find int RVs
        sbart = RV_Retrieval(args.snr,obs.training[0,0][non_ones[0][0]:non_ones[0][-1]+1],obs.wgrid,obs.inst_wgrid,args.N,type='sample')
        intrvs, uncs = sbart.find_dv(synthetic_spectra[:,:].cpu(),uncertainty.cpu()[:,:],obs.berv[:].cpu(),func='connors')

        def f_wrapped(x):
            return forward_model(x, obs.wgrid, obs.inst_wgrid, obs.berv.unsqueeze(0), obs.planet.unsqueeze(0), change_res)
        x = obs.training[:,:,non_ones[0][0]:non_ones[0][-1]+1]
        A_full = jacobian(f_wrapped, x, create_graph=True)
        A = A_full[0,:,:,0,0,:]
        AtA =torch.matmul(A, A.transpose(-1, -2)).contiguous()

        # Initialize
        B = args.B
        bervs_to_send = obs.berv.unsqueeze(0).expand(B, args.N)
        planetrv_for_spectrum_sample = torch.tensor(templatervs).to(device).unsqueeze(0).expand(B, args.N)

        spectrum_samples = []
        RV_samples = []

        for gibb in tqdm(range(args.gibbs_steps)):
            # Spectrum Sample
            LSF = Score_Likelihood(Y=synthetic_spectra, V=planetrv_for_spectrum_sample, sig_n=uncertainty,
                                   spec_wgrid=obs.wgrid, inst_wgrid=obs.inst_wgrid, non_ones=non_ones[0], SNR=args.snr,
                                   berv=bervs_to_send, beta_min=1e-2, beta_max=20, AtA=AtA, change_res=change_res)
            dimensions = [1, len(obs.padded_wgrid)]
            posterior_samples = model.sample(shape=[B, *dimensions], steps=3000, likelihood_score_fn=LSF)
            posterior_samples_np = posterior_samples.cpu()

            # MALA Sample
            mala = MALA(synthetic_spectra[:,:], uncertainty[:,:], obs.berv, args.snr, obs.inst_wgrid, obs.wgrid)
            samples, accepted = mala.find_rv(planetrv_for_spectrum_sample, posterior_samples[:,:,non_ones[0][0]:non_ones[0][-1]+1], 2000)

            planetrv_for_spectrum_sample = torch.mean(samples[500:,:,:], dim=0)

            spectrum_samples.append(posterior_samples_np.numpy())
            RV_samples.append(samples[500:,:,:].cpu().numpy())

        # Final MALA sample
        mala = MALA(synthetic_spectra[:,:], uncertainty[:,:], obs.berv, args.snr, obs.inst_wgrid, obs.wgrid)
        true_samples, accepted = mala.find_rv(torch.tensor(templatervs).to(device).unsqueeze(0).expand(B, args.N),
                                              obs.training.repeat(B,1,1)[:,:,non_ones[0][0]:non_ones[0][-1]+1], 5000)

        columns = ['posterior_spectrum','RV_samples','true_planet','true_spectrum',
                   'observations','uncertainty','snr','template','template_rvs', 'true_recovered_rvs',
                   'inst_wgrid','wgrid',"true_spec_mala_rv",'berv','dV','non_ones']
        values = [np.array(spectrum_samples),
                  np.array(RV_samples),
                  obs.planet.cpu().numpy(),
                  obs.training.cpu().numpy(),
                  synthetic_spectra.cpu().numpy(),
                  uncertainty.cpu().numpy(),
                  args.snr,
                  template.cpu().numpy(),
                  templatervs,
                  intrvs,
                  obs.inst_wgrid.cpu().numpy(),
                  obs.wgrid.cpu().numpy(),
                  true_samples.cpu().numpy(),
                  obs.berv.cpu().numpy(),
                  mala.deltaV,
                  non_ones]

        # Store in group named after seed
        seed_group = f_out.create_group(f"seed_{seed}")
        for col, val in zip(columns, values):
            seed_group.create_dataset(col, data=val)
