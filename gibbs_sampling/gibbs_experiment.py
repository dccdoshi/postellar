import sys
sys.path.append("../src")
import numpy as np
import torch
import h5py
from tqdm import tqdm
from torch.autograd.functional import jacobian
import argparse
import itertools

from synthetic_observations import Observations
from transformer import *
from template import Template
from sbart_rv_finder import RV_Retrieval
from score_models import ScoreModel
from spectrum_lsf import Score_Likelihood
from mala import MALA

# -------------------
# Command-line arguments
# -------------------
parser = argparse.ArgumentParser(description="Run Gibbs + MALA pipeline")
parser.add_argument("-i", type=int, nargs="+", required=True, help="List of index values")
parser.add_argument("-snr", type=int, nargs="+", required=True, help="List of SNR values")
parser.add_argument("-ntemp", type=int, nargs="+", required=True, help="List of nspec values")
parser.add_argument("-order",type=int, required=True,help="Order number")
parser.add_argument('-val', type=str, default='SPIRou20_val.df', help='model directory')
parser.add_argument('-model', type=str, default='b16nf16ch2_4_spir20_e500', help='model directory')
parser.add_argument('-output', type=str, default='output.h5', help='HDF5 file to save data')
parser.add_argument('-gibb', type=int, default=1, help='gibb_step')
parser.add_argument('-step', type=int, default=3000,nargs="+", help='sampler_step')

args = parser.parse_args()

# -------------------
# Build parameter combos
# -------------------
parameters = list(itertools.product(args.i, args.snr, args.ntemp))
output_file = args.output

# -------------------
# Hardcoded / defaults
# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_data_file = args.val
num_seeds = 10
gibbs_steps = args.gibb
checkpoints_directory = "../../order_model/"+args.model
B = 5

# -------------------
# Load model once
# -------------------
model = ScoreModel(checkpoints_directory=checkpoints_directory, device=device)
change_res = True


# -------------------
# Iterate through parameter comboniations Open HDF5 for writing
# -------------------

for param_idx, (i, snr, nspec) in enumerate(parameters):
    print(f"\n===== Running parameter combo {param_idx} | i={i}, SNR={snr}, nspec={nspec}=====", flush=True)

    steps = 3_000
    # if snr == 50:
    #     steps = 5_000
    # elif snr== 100:
    #     steps = 7_500
    if snr>100:
        steps = 10_000
    elif snr>50:
        steps = 5_000
    print(steps)
    # unique file name for each parameter combo
    filename = f"results/{output_file}_i{i}_snr{snr}nspec{nspec}.h5"

    with h5py.File(filename, "w") as f_out:
        # -------------------
        # Create A-group in HDF5
        # -------------------
        obs = Observations(i=0, seed=0, SNR=100, filepath="../data/"+val_data_file, N=5,order=args.order)
        non_ones = torch.where(obs.padded_wgrid != 1)
        group_A = f_out.create_group("Order")
        group_A.attrs["order"] = args.order
        group_A.create_dataset("wgrid", data=obs.wgrid.cpu().numpy())
        group_A.create_dataset("inst_wgrid", data=obs.inst_wgrid.cpu().numpy())
        group_A.create_dataset("non_ones", data=non_ones[0].cpu().numpy())
        
        # -------------------
        # Create observations
        # -------------------
        obs = Observations(i=i, seed=0, SNR=snr, filepath="../data/"+val_data_file, N=nspec,order=args.order)
        synthetic_spectra, uncertainty = obs.make_observations(func='connors', add_RV=True, change_res=change_res)
        synthetic_spectra, uncertainty = obs.post_process()
        non_ones = torch.where(obs.padded_wgrid != 1)
        
        # -------------------
        # Create template
        # -------------------
        temp = Template(synthetic_spectra, obs.berv, obs.inst_wgrid, obs.wgrid)
        template = temp.make_template(func='scipy')
        
        # -------------------
        # Initialize RVs for Gibbs
        # -------------------
        sbart_init = RV_Retrieval(snr, template, obs.wgrid, obs.inst_wgrid, nspec)
        templatervs_init, template_unc_init = sbart_init.find_dv(synthetic_spectra.cpu(), uncertainty.cpu(), obs.berv.cpu(), func='connors')

        # -------------------
        # Initialize RV for Gibbs
        # -------------------
        bervs_to_send = obs.berv.unsqueeze(0).expand(B, nspec)
        planetrv_forA = torch.tensor(templatervs_init).to(device).unsqueeze(0)
        planetrv_for_spectrum_sample = torch.tensor(templatervs_init).to(device).unsqueeze(0).expand(B, nspec)
        
        # -------------------
        # Calculate AtA for Gibbs
        # -------------------
        list_AtA = []
        for planet_chunk, berv_chunk in zip(obs.planet, obs.berv):
            planetrv_forA = torch.as_tensor(planet_chunk, device=device).unsqueeze(0).unsqueeze(0)
            bervy = berv_chunk.unsqueeze(0).unsqueeze(0)

            def f_wrapped(x):
                return forward_model(x, obs.wgrid, obs.inst_wgrid, bervy, planetrv_forA, change_res)

            x = obs.training[:, :, non_ones[0]]
            A_full = jacobian(f_wrapped, x, create_graph=False)
            A = A_full[0, :, :, 0, 0, :]                 # [chunk, L, L]
            chunk_AtA = torch.matmul(A, A.transpose(-1, -2))   # [chunk, L, L]

            list_AtA.append(chunk_AtA)
            del A_full, A, chunk_AtA
            torch.cuda.empty_cache()

        AtA = torch.cat(list_AtA)
        list_AtA = []
        
        # -------------------
        # Gibbs Sampling
        # -------------------
        
        spectrum_samples = []
        for gibb in tqdm(range(gibbs_steps)):
            LSF = Score_Likelihood(Y=synthetic_spectra, V=planetrv_for_spectrum_sample, sig_n=uncertainty,
                                    spec_wgrid=obs.wgrid, inst_wgrid=obs.inst_wgrid, non_ones=non_ones[0], SNR=snr,
                                    berv=bervs_to_send, beta_min=1e-2, beta_max=20, AtA=AtA, change_res=change_res)
            dimensions = [1, len(obs.padded_wgrid)]
            posterior_samples = model.sample(shape=[B, *dimensions], steps=steps, likelihood_score_fn=LSF)
            
            mala = MALA(synthetic_spectra, uncertainty, obs.berv, snr, obs.inst_wgrid, obs.wgrid)
            print(planetrv_for_spectrum_sample.shape, posterior_samples[:, :, non_ones[0][0]:non_ones[0][-1]+1].shape)
            samples, accepted = mala.find_rv(planetrv_for_spectrum_sample, posterior_samples[:, :, non_ones[0][0]:non_ones[0][-1]+1], 1000)
            planetrv_for_spectrum_sample = torch.mean(samples[500:], dim=0)
            spectrum_samples.append(posterior_samples)
        del AtA
        torch.cuda.empty_cache()
        spectrum_samples = torch.stack(spectrum_samples,dim=0)
        print(spectrum_samples.shape)
        mean_spectrum_sample = spectrum_samples.mean(dim=(0),keepdim=True)[0]
        print(mean_spectrum_sample.shape)
        # -------------------
        # Create B-group
        # -------------------
        group_B = group_A.create_group("Observational Parameters")
        group_B.attrs["i"] = i
        group_B.attrs["snr"] = snr
        group_B.attrs["nspec"] = nspec
        group_B.attrs["deltaV"] = mala.deltaV.cpu()
        
        # -------------------
        # Create C-group
        # -------------------
        group_C = group_B.create_group("Spectrum")
        group_C.create_dataset("posterior_spectrum_samples", data=spectrum_samples.cpu().numpy())
        group_C.create_dataset("template", data=template.cpu().numpy())
        group_C.create_dataset("true_spectrum", data=obs.training.cpu().numpy())
        
        # -------------------
        # Create D-group (per seed × true planet value)
        # -------------------
        group_D = group_B.create_group("RV Samples")
        for seed in range(num_seeds):
            print("At this seed"+str(seed),flush=True)
            # Create evaluation obs with new planet value
            eval_obs = Observations(i=i, seed=seed, SNR=snr, filepath="../data/"+val_data_file, N=100,order=args.order)
            eval_spectra, eval_unc = eval_obs.make_observations(func='connors', add_RV=True, change_res=change_res)
            eval_spectra, eval_unc = eval_obs.post_process()
            
            # Template RVs
            sbart_eval = RV_Retrieval(snr, template, eval_obs.wgrid, eval_obs.inst_wgrid, nspec)
            temp_rv, temp_unc = sbart_eval.find_dv(eval_spectra.cpu(), eval_unc.cpu(), eval_obs.berv.cpu(), func='connors')
            print("Done template",flush=True)
            # Intrinsic RVs
            sbart_int = RV_Retrieval(snr, eval_obs.training[:, :, non_ones[0][0]:non_ones[0][-1]+1], eval_obs.wgrid, eval_obs.inst_wgrid, 0, type='sample')
            int_rv, int_unc = sbart_int.find_dv(eval_spectra.cpu(), eval_unc.cpu(), eval_obs.berv.cpu(), func='connors')
            print("Done int",flush=True)

            # bart propr RVs
            sbart_prior = RV_Retrieval(snr, mean_spectrum_sample.mean(dim=(0),keepdim=True)[:, :, non_ones[0][0]:non_ones[0][-1]+1], eval_obs.wgrid, eval_obs.inst_wgrid, 0, type='sample')
            prior_rv, prior_unc = sbart_prior.find_dv(eval_spectra.cpu(), eval_unc.cpu(), eval_obs.berv.cpu(), func='connors')
            print("Done bart prior",flush=True)

            # MALA using mean spectrum sample
            planetrv_for_spectrum_sample = torch.tensor(temp_rv).to(device).unsqueeze(0).expand(B, 100)
            mala_eval = MALA(eval_spectra, eval_unc, eval_obs.berv, snr, eval_obs.inst_wgrid, eval_obs.wgrid)
            print(planetrv_for_spectrum_sample.shape, mean_spectrum_sample[:, :, non_ones[0][0]:non_ones[0][-1]+1].shape)
            mala_samples, _ = mala_eval.find_rv(planetrv_for_spectrum_sample,
                                                mean_spectrum_sample[:, :, non_ones[0][0]:non_ones[0][-1]+1], 1000)
            print("Done MALA",flush=True)
            # Save to HDF5
            group_seed = group_D.create_group(f"seed_{seed}")
            group_seed.create_dataset("true_planet", data=eval_obs.planet.cpu().numpy())
            group_seed.create_dataset("mala_samples", data=mala_samples[100:].cpu().numpy())
            group_seed.create_dataset("template_rv", data=temp_rv)
            group_seed.create_dataset("template_uncertainty", data=temp_unc)
            group_seed.create_dataset("intrinsic_rv", data=int_rv)
            group_seed.create_dataset("intrinsic_uncertainty", data=int_unc)
            group_seed.create_dataset("prior_rv", data=prior_rv)
            group_seed.create_dataset("prior_uncertainty", data=prior_unc)

print("\nAll parameter combos processed. Results saved to results/output.h5")
