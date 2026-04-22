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
# This gives us access to all the parameters we wanted to test for a given order analysis.
# -------------------
parser = argparse.ArgumentParser(description="Run Gibbs + MALA pipeline")
parser.add_argument("-i", type=int, nargs="+", required=True, help="This provides the index number ie which PHOENIX spectrum you want to use as the base for your observations. \
                                                                    This index number relates to the index for the validation dataframes. Therefore if you want to test over a specific PHOENIX model, \
                                                                    open the dataframe, find the model with the parameters you want, use that specific index here")

parser.add_argument("-snr", type=int, nargs="+", required=True, help="List of SNR values you wanted to test for")
parser.add_argument("-ntemp", type=int, nargs="+", required=True, help="List of Ntemp values. Ntemp refers to how many observations do you want to use to inform your spectrum inference.")
parser.add_argument("-order",type=int, required=True,help="Order number: What order are we doing this analysis over")
parser.add_argument('-val', type=str, default='SPIRou20_val.df', help='This should reference the validation file with the given order.')
parser.add_argument('-model', type=str, default='b16nf16ch2_4_spir20_e500', help='This should reference the trained ML model with the given order')
parser.add_argument('-output', type=str, default='output.h5', help='HDF5 file to save outputs')
parser.add_argument('-gibb', type=int, default=1, help='The number of gibbs steps. This should just be set to one.')
parser.add_argument('-step', type=int, default=3000,nargs="+", help='The number of Euler-Maryuma steps for the spectrum sampling.')

args = parser.parse_args()

# -------------------
# Build parameter combos
# This simply creates the parameter combos, such that we analyze each parameter combo, save its outputs, and then move onto the next parameter combo
# -------------------
parameters = list(itertools.product(args.i, args.snr, args.ntemp))
output_file = args.output

# -------------------
# Hardcoded / defaults
# DO NOT CHANGE THESE AT ALL
# -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_data_file = args.val
gibbs_steps = args.gibb
checkpoints_directory = "../../order_model/"+args.model
bmin= 1e-2
bmax = 20

'''
 Defaults you can change
'''
# How many noise instances do you want to test over when we test how well our spectrum is at retrieving RVs
num_seeds = 5
# How many posterior spectra do we want to sample?
B = 5

# -------------------
# Load model once
# -------------------
model = ScoreModel(checkpoints_directory=checkpoints_directory, device=device)


# -------------------
# Iterate through parameter comboniations Open HDF5 for writing
# -------------------

for param_idx, (i, snr, nspec) in enumerate(parameters):
    print(f"\n===== Running parameter combo {param_idx} | i={i}, SNR={snr}, nspec={nspec}=====", flush=True)

    '''
    Redetermine number of Euler-Maryuma Steps
    '''
    steps = 3_000
    # We have to increase the number of steps if the SNR is high, because the likelihood becomes constraining in this limit 
    # Thus we need smaller steps such that we don't overshoot from the likelihood
    if snr>100:
        steps = 10_000
    elif snr>50:
        steps = 5_000
    print("These are the number of Euler-Maryuma steps for the spectrum sampler: ",steps,flush=True)


    # unique file name for each parameter combo
    filename = f"results/{output_file}_i{i}_snr{snr}nspec{nspec}.h5"

    with h5py.File(filename, "w") as f_out:
        # -------------------
        # Create A-group in HDF5
        # This is simply done to save the key parameters from this test
        # -------------------
        obs = Observations(i=0, seed=0, SNR=100, filepath="../data/validation_data/"+val_data_file, N=5,order=args.order)
        non_ones = torch.where(obs.padded_wgrid != 1)
        group_A = f_out.create_group("Order")
        group_A.attrs["order"] = args.order
        # This is the native wavelength grid before any transformation function is applied (the wavelength grid of the samples)
        group_A.create_dataset("wgrid", data=obs.wgrid.cpu().numpy())
        # This is the wavelength grid of the synthetic data
        group_A.create_dataset("inst_wgrid", data=obs.inst_wgrid.cpu().numpy())
        # This captures the padding that we had to include in the samples (the padding is subsequently removed when we compare samples to data)
        group_A.create_dataset("non_ones", data=non_ones[0].cpu().numpy())
        
        # -------------------
        # Create synthetic observations
        # -------------------
        obs = Observations(i=i, seed=0, SNR=snr, filepath="../data/validation_data/"+val_data_file, N=nspec,order=args.order)
        synthetic_spectra, uncertainty = obs.make_observations(func='connors', add_RV=True)
        synthetic_spectra, uncertainty = obs.post_process()
        non_ones = torch.where(obs.padded_wgrid != 1)
        print("I have created the synthetic observations.",flush=True)

        # -------------------
        # Create template
        # -------------------
        temp = Template(synthetic_spectra, obs.berv, obs.inst_wgrid, obs.wgrid)
        template = temp.make_template(func='scipy')
        print("I have created the template.",flush=True)
        
        # -------------------
        # Find initial RVs using template-matching
        # -------------------
        sbart_init = RV_Retrieval(snr, template, obs.wgrid, obs.inst_wgrid, nspec)
        templatervs_init, template_unc_init = sbart_init.find_dv(synthetic_spectra.cpu(), uncertainty.cpu(), obs.berv.cpu(), func='connors')

        # -------------------
        # Restructure the retrieved RVs to fit the tensor shapes needed for later analysis
        # -------------------
        bervs_for_sampling = obs.berv.unsqueeze(0).expand(B, nspec)
        planetrv_for_spectrum_sample = torch.tensor(templatervs_init).to(device).unsqueeze(0).expand(B, nspec)
        
        # -------------------
        # Calculate AtA for spectrum sampling 
        # This is a necessary step to do posterior sampling. This is an extremely computationally heavy step. 
        # Work needs to be done to figure out how save this matrix more efficiently. 
        # TLDR this captures the transformation matrix "A" for each observation (berv value) and saves them in a torch list
        # -------------------
        list_AtA = []
        for planet_chunk, berv_chunk in zip(obs.planet, obs.berv):
            planetrv_for_A = torch.as_tensor(planet_chunk, device=device).unsqueeze(0).unsqueeze(0)
            berv_for_A = berv_chunk.unsqueeze(0).unsqueeze(0)

            def f_wrapped(x):
                return forward_model(x, obs.wgrid, obs.inst_wgrid, berv_for_A, planetrv_for_A)

            x = obs.training[:, :, non_ones[0]]
            A_full = jacobian(f_wrapped, x, create_graph=False)
            A = A_full[0, :, :, 0, 0, :]                 # [chunk, L, L]
            chunk_AtA = torch.matmul(A, A.transpose(-1, -2))   # [chunk, L, L]

            list_AtA.append(chunk_AtA)
            del A_full, A, chunk_AtA
            torch.cuda.empty_cache()

        AtA = torch.cat(list_AtA)
        list_AtA = []
        print("I have created the AtA matrix.",flush=True)
       
        # -------------------
        # Gibbs Sampling
        # Here we sample for the spectrum first with our synthetic observations
        # Then we sample for our RVs. The RVs are not really as important here. 
        # -------------------
        
        spectrum_samples = []
        for gibb in tqdm(range(gibbs_steps)):
            print("I am starting to sample for the spectrum.",flush=True)

            # This defines the likelihood function that our posterior sampler will use to do posterior sampling
            LSF = Score_Likelihood(Y=synthetic_spectra, V=planetrv_for_spectrum_sample, sig_n=uncertainty,
                                    spec_wgrid=obs.wgrid, inst_wgrid=obs.inst_wgrid, non_ones=non_ones[0], SNR=snr,
                                    berv=bervs_for_sampling, beta_min=bmin, beta_max=bmax, AtA=AtA)
            dimensions = [1, len(obs.padded_wgrid)]

            # This actually does the posterior spectrum sampling 
            posterior_samples = model.sample(shape=[B, *dimensions], steps=steps, likelihood_score_fn=LSF)
            
            # This sets up the RV sampler
            mala = MALA(synthetic_spectra, uncertainty, obs.berv, snr, obs.inst_wgrid, obs.wgrid)
            
            # This actually does the RV sampling
            samples, accepted = mala.find_rv(planetrv_for_spectrum_sample, posterior_samples[:, :, non_ones[0][0]:non_ones[0][-1]+1], steps=1000)

            # Save the samples we generated. Take out the burn in phase for the RV samples
            planetrv_for_spectrum_sample = torch.mean(samples[500:], dim=0)
            spectrum_samples.append(posterior_samples)

        del AtA
        torch.cuda.empty_cache()
        spectrum_samples = torch.stack(spectrum_samples,dim=0)
        mean_spectrum_sample = spectrum_samples.mean(dim=(0),keepdim=True)[0]
        print("I am done sampling for the spectrum.",flush=True)

        # -------------------
        # Create B-group
        # Saving the parameter combo values here. 
        # -------------------
        group_B = group_A.create_group("Observational Parameters")
        group_B.attrs["i"] = i
        group_B.attrs["snr"] = snr
        group_B.attrs["nspec"] = nspec
        group_B.attrs["deltaV"] = mala.deltaV.cpu()
        
        # -------------------
        # Create C-group
        # Saving the relevant spectrum outputs here
        # -------------------
        group_C = group_B.create_group("Spectrum")
        group_C.create_dataset("posterior_spectrum_samples", data=spectrum_samples.cpu().numpy())
        group_C.create_dataset("template", data=template.cpu().numpy())
        group_C.create_dataset("true_spectrum", data=obs.training.cpu().numpy())
        
        # -------------------
        # Create D-group (per seed × true planet value)
        # Here we want to test our new spectrum in doing RV analysis. So we create a bunch of different observations with different noise instances 'seeds' 
        # and evaluate over these observations
        # -------------------
        print("I am going to sample the RVs now.",flush=True)

        group_D = group_B.create_group("RV Samples")
        for seed in range(num_seeds):
            print("At this seed"+str(seed),flush=True)
            # Create evaluation observations with new planet value
            eval_obs = Observations(i=i, seed=seed, SNR=snr, filepath="../data/validation_data/"+val_data_file, N=100,order=args.order)
            eval_spectra, eval_unc = eval_obs.make_observations(func='connors', add_RV=True)
            eval_spectra, eval_unc = eval_obs.post_process()
            
            # Template RVs
            sbart_eval = RV_Retrieval(snr, template, eval_obs.wgrid, eval_obs.inst_wgrid, nspec)
            temp_rv, temp_unc = sbart_eval.find_dv(eval_spectra.cpu(), eval_unc.cpu(), eval_obs.berv.cpu(), func='connors')
            print("Done template",flush=True)

            # Intrinsic RVs
            sbart_int = RV_Retrieval(snr, eval_obs.training[:, :, non_ones[0][0]:non_ones[0][-1]+1], eval_obs.wgrid, eval_obs.inst_wgrid, 0, type='sample')
            int_rv, int_unc = sbart_int.find_dv(eval_spectra.cpu(), eval_unc.cpu(), eval_obs.berv.cpu(), func='connors')
            print("Done int",flush=True)

            # Use the mean spectrum sample with the SBART template matching technique
            sbart_prior = RV_Retrieval(snr, mean_spectrum_sample.mean(dim=(0),keepdim=True)[:, :, non_ones[0][0]:non_ones[0][-1]+1], eval_obs.wgrid, eval_obs.inst_wgrid, 0, type='sample')
            prior_rv, prior_unc = sbart_prior.find_dv(eval_spectra.cpu(), eval_unc.cpu(), eval_obs.berv.cpu(), func='connors')
            print("Done bart prior",flush=True)

            # MALA using your spectrum samples
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
