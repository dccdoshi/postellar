"""
Full pipeline:
  - Build final_results from HDF5 files in `results_folder`
  - For Mp in Mp_list:
      - generate true RVs
      - augment RVs using final_results (mala/template/intrinsic)
      - for each realization, write temp .dat, run juliet to fit K_p1, extract posterior median +/-1sigma
      - clean up and save summary CSV

Run-time notes:
 - This will run many short juliet fits; set n_live_points lower for tests.
 - Requires: numpy, pandas, h5py, juliet, tqdm (optional)
"""

import os
import numpy as np
import h5py
import tempfile
import shutil
import pandas as pd
import juliet
from tqdm import tqdm

# ---------------------------
# PART A: Build final_results from HDF5 folder
# ---------------------------
def build_final_results_from_h5(folder, pattern_prefix='order_29'):
    """
    Read HDF5 files in `folder` whose filenames start with `pattern_prefix` and build final_results.

    Returns: final_results dict keyed by (i, snr, nspec) with keys:
      'intrinsic_z','template_z','mala_z','template_unc','intrinsic_unc','mala_unc'
    Each value will be a list/np.array of realizations (1D arrays) suitable for later reshaping.
    """
    all_data = {}

    for filename in os.listdir(folder):
        if not filename.startswith(pattern_prefix):
            continue
        filepath = os.path.join(folder, filename)
        print("Reading:", filepath)
        with h5py.File(filepath, "r") as f:
            # Read observational params (guarded)
            try:
                i = int(f['Order']['Observational Parameters'].attrs['i'])
                snr = int(f['Order']['Observational Parameters'].attrs['snr'])
                nspec = int(f['Order']['Observational Parameters'].attrs['nspec'])
            except Exception as e:
                print(f"Skipping {filename}: missing observational attributes ({e})")
                continue

            # RV Samples group path (guarded)
            try:
                rv_group = f['Order']['Observational Parameters']['RV Samples']
            except KeyError:
                print(f"No RV Samples group in {filename}; skipping.")
                continue

            # Gather per-seed dictionaries
            rv_analysis = {}
            for seed in rv_group.keys():
                seed_group = rv_group[seed]
                # For each expected dataset, convert to numpy arrays (if present)
                # Use .value-like access via [:] or [()]
                seed_dict = {}
                for subname in seed_group.keys():
                    try:
                        seed_dict[subname] = seed_group[subname][()]
                    except Exception:
                        # fallback to empty array
                        seed_dict[subname] = np.array([])
                rv_analysis[seed] = seed_dict

            # Store under key
            all_data[(i, snr, nspec)] = {"rv": rv_analysis}

    # Now process all_data into the condensed final_results format
    final_results = {}
    for key, data in all_data.items():
        i, snr, nspec = key
        rv_analysis = data["rv"]

        # Prepare lists to accumulate across seeds
        results = {
            "intrinsic_z": [],
            "template_z": [],
            "mala_z": [],
            "template_unc": [],
            "intrinsic_unc": [],
            "mala_unc": []
        }

        for seed, group in rv_analysis.items():
            # Ensure fields exist. If not, skip this seed.
            # expected keys in group: 'true_planet', 'intrinsic_rv', 'template_rv',
            # 'mala_samples', 'intrinsic_uncertainty', 'template_uncertainty', 'mala_uncertainty' (maybe)
            if 'true_planet' not in group:
                # skip incomplete seed
                continue

            true = np.array(group.get("true_planet", []))
            intrinsic = np.array(group.get("intrinsic_rv", []))
            template = np.array(group.get("template_rv", []))
            mala_mean = np.array(group.get("mala_rv", []))
            # mala = np.array(group.get("mala_samples", []))  # could be (n_samples, N) or similar
            # Compute mala point-estimates (means/stds) in a robust manner
            # If mala is 2D or 3D, collapse first axes to get per-observation arrays
            # if mala.size == 0:
            #     continue

            # # Compute mean and std across sample axes (collapse all except last axis if possible)
            # if mala.ndim == 1:
            #     mala_mean = mala
            #     mala_std = np.ones_like(mala) * np.std(mala)
            # else:
            #     # collapse leading axes except last axis length equals number of observations
            #     # common shapes: (n_samples, N) or (n_chains, n_samples, N)
            #     mala_mean = np.median(mala, axis=tuple(range(mala.ndim - 1)))
            #     mala_std = np.std(mala, axis=tuple(range(mala.ndim - 1)))

            # Uncertainties (these may be arrays per-observation)
            intrinsic_unc = np.array(group.get("intrinsic_uncertainty", np.full_like(true, np.nan)))
            template_unc = np.array(group.get("template_uncertainty", np.full_like(true, np.nan)))
            # If a 'mala_uncertainty' exists use it, otherwise use mala_std
            mala_std = np.array(group.get("mala_uncertainty", np.full_like(true, np.nan)))

            # Compute z-scores only if shapes align
            try:
                intrinsic_z = (intrinsic - true) / intrinsic_unc
            except Exception:
                intrinsic_z = np.full_like(true, np.nan)

            try:
                template_z = (template - true) / template_unc
            except Exception:
                template_z = np.full_like(true, np.nan)

            try:
                mala_z = (mala_mean - true) / mala_std
            except Exception:
                mala_z = np.full_like(true, np.nan)

            # Filter extreme outliers in mala_z (same heuristic you had)
            mask = np.isfinite(mala_z) & (mala_z < 1e10) & (mala_z > -1e10)
            mala_z = mala_z[mask]
            mala_unc_filtered = mala_std[mask] if mala_std.shape == mala_z.shape else mala_std

            # Append to results lists
            results["template_unc"].append(np.array(template_unc))
            results["intrinsic_unc"].append(np.array(intrinsic_unc))
            results["mala_unc"].append(np.array(mala_unc_filtered))
            results["intrinsic_z"].append(np.array(intrinsic_z))
            results["template_z"].append(np.array(template_z))
            results["mala_z"].append(np.array(mala_z))

        # Convert lists to numpy arrays for consistent downstream handling
        # Keep as-list-of-arrays where each entry is a realization vector
        final_results[(i, snr, nspec)] = {k: v for k, v in results.items()}

    return final_results

# ---------------------------
# PART B: RV generation, augmentation, juliet fit utilities (from previous script)
# ---------------------------
def get_K(P_days, Mp_jupiter, Ms_solar):
    return 28.4 * ((P_days / 365.2422) ** (-1/3)) * Mp_jupiter * (Ms_solar ** (-2/3))

def get_RV(P_days, Mp_jupiter, Ms_solar, N=20, t0=2460988.218692):
    n_dense = N // 4  # number of dense points per cluster
    n_sparse = N - 2 * n_dense -2

    # Uniform sparse coverage across full orbit
    sparse_phases = np.linspace(0, 1, n_sparse, endpoint=False)

    # Denser sampling around peaks/troughs
    peak_phases = np.linspace(0.20, 0.30, n_dense)
    trough_phases = np.linspace(0.70, 0.80, n_dense)
    fixed_phases = np.array([0.25,0.75])

    # Combine and wrap to [0, 1)
    phases = np.sort(np.mod(np.concatenate([sparse_phases, peak_phases, trough_phases, fixed_phases]), 1.0))

    # Convert to times
    dates = t0 + phases * P_days
    # dates = np.linspace(t0, t0 + P_days, N)
    phase = 2 * np.pi * ((dates - t0) / P_days)
    K = get_K(P_days, Mp_jupiter, Ms_solar)
    RVs = K * np.cos(phase + (np.pi / 2.0))
    return dates, RVs, K

def get_uncertainty(final_results, i=0, snr=25, nspec=10):
    fr = final_results[(i, snr, nspec)]
    # fr fields are lists of arrays; take mean of per-realization uncertainties then divide sqrt(50) like you did
    # Handle case where lists are empty
    if len(fr['template_unc']) == 0:
        raise RuntimeError("No template_unc found for this key in final_results")
    template_unc = np.mean([np.mean(x) for x in fr['template_unc']]) / np.sqrt(1.1)
    mala_unc = np.mean([np.mean(x) for x in fr['mala_unc']]) / np.sqrt(1.1)
    intrinsic_unc = np.mean([np.mean(x) for x in fr['intrinsic_unc']]) / np.sqrt(1.1)
    return template_unc, mala_unc, intrinsic_unc

def get_augmented_RVs(RVs, final_results, N=20, i=0, snr=25, nspec=10):
    """
    Returns:
      mala_RVs, template_RVs, int_RVs : arrays shaped (n_realizations, N)
      uncs : tuple (mala_unc, template_unc, int_unc) scalars
    Each row in the *_RVs arrays is one realization (RV series).
    """
    fr = final_results[(i, snr, nspec)]

    # Convert lists-of-arrays to 2D arrays where each row is a realization
    mala_z_list = [np.asarray(x).reshape(-1) for x in fr['mala_z'] if np.asarray(x).size > 0]
    template_z_list = [np.asarray(x).reshape(-1) for x in fr['template_z'] if np.asarray(x).size > 0]
    intrinsic_z_list = [np.asarray(x).reshape(-1) for x in fr['intrinsic_z'] if np.asarray(x).size > 0]

    if len(mala_z_list) == 0 and len(template_z_list) == 0 and len(intrinsic_z_list) == 0:
        raise RuntimeError("No z-score realizations found in final_results for the given key")

    mala_z_arr = np.vstack(np.random.permutation(mala_z_list)) if len(mala_z_list) > 0 else np.zeros((0, N))
    template_z_arr = np.vstack(np.random.permutation(template_z_list)) if len(template_z_list) > 0 else np.zeros((0, N))
    intrinsic_z_arr = np.vstack(np.random.permutation(intrinsic_z_list)) if len(intrinsic_z_list) > 0 else np.zeros((0, N))

    # Scalar uncertainties
    template_unc, mala_unc, int_unc = get_uncertainty(final_results, i, snr, nspec)

    # Convert z-scores → offsets by multiplying by scalar uncertainties
    mala_offsets = (mala_z_arr * mala_unc).reshape(-1, N) if mala_z_arr.size else np.zeros((0, N))
    template_offsets = (template_z_arr * template_unc).reshape(-1, N) if template_z_arr.size else np.zeros((0, N))
    intrinsic_offsets = (intrinsic_z_arr * int_unc).reshape(-1, N) if intrinsic_z_arr.size else np.zeros((0, N))

    # Add true RVs (broadcast)
    mala_RVs = mala_offsets + RVs
    template_RVs = template_offsets + RVs
    int_RVs = intrinsic_offsets + RVs

    return mala_RVs, template_RVs, int_RVs, (mala_unc, template_unc, int_unc)

def make_priors_for_K_only(P_days, t0, K_bounds=(-100.0, 100.0)):
    priors = {}
    priors['K_p1'] = {'distribution': 'uniform', 'hyperparameters': list(K_bounds)}
    priors['P_p1'] = {'distribution': 'fixed', 'hyperparameters': float(P_days)}
    priors['t0_p1'] = {'distribution': 'fixed', 'hyperparameters': float(t0)}
    priors['ecc_p1'] = {'distribution': 'fixed', 'hyperparameters': 0.0}
    priors['omega_p1'] = {'distribution': 'fixed', 'hyperparameters': 0.0}
    for inst in ['CORALIE14', 'CORALIE07', 'HARPS', 'FEROS']:
        priors[f'mu_{inst}'] = {'distribution': 'fixed', 'hyperparameters': 0.0}
        priors[f'sigma_w_{inst}'] = {'distribution': 'fixed', 'hyperparameters': 0.0}
    return priors

def write_temp_dat(filename, times, rvs, rvs_err, instrument_name='HARPS'):
    inst_col = np.array([instrument_name] * len(times))
    data_to_save = np.column_stack([times, rvs, rvs_err, inst_col])
    np.savetxt(filename, data_to_save, fmt='%s', comments='')

def extract_K_samples(results):
    candidates = []
    if hasattr(results, 'posterior_samples'):
        try:
            if 'K_p1' in results.posterior_samples:
                candidates.append(results.posterior_samples['K_p1'])
        except Exception:
            pass
    if hasattr(results, 'posteriors'):
        try:
            if isinstance(results.posteriors, dict):
                if 'K_p1' in results.posteriors:
                    candidates.append(np.array(results.posteriors['K_p1']).ravel())
                if 'posterior_samples' in results.posteriors and 'K_p1' in results.posteriors['posterior_samples']:
                    candidates.append(np.array(results.posteriors['posterior_samples']['K_p1']).ravel())
        except Exception:
            pass
    try:
        # Some juliet wrappers keep data under results.result.posterior_samples
        ps = getattr(getattr(results, 'result', None), 'posterior_samples', None)
        if ps is not None and 'K_p1' in ps:
            candidates.append(np.array(ps['K_p1']).ravel())
    except Exception:
        pass

    for c in candidates:
        try:
            arr = np.asarray(c).ravel()
            if arr.size > 0:
                return arr
        except Exception:
            continue
    # If not found, dump some debug info to help user inspect:
    raise RuntimeError(f"Could not find K_p1 posterior samples; inspect your `results` object. Available attrs: {dir(results)}")

# ---------------------------
# PART C: Main run function (wraps everything)
# ---------------------------
def run_full_pipeline(results_folder,
                      pattern_prefix='order_29',
                      Mp_list=[0.01],
                      P_days=7.0,
                      Ms_solar=0.12,
                      t0=2460988.218692,
                      N=20,
                      i=0, snr=25, nspec=10,
                      nreal = 10,
                      n_live_points=50,
                      out_csv='k_results_summary.csv'):
    # Build final_results from HDF5 files
    final_results = build_final_results_from_h5(results_folder, pattern_prefix=pattern_prefix)
    if (i, snr, nspec) not in final_results:
        raise RuntimeError(f"Key {(i,snr,nspec)} not found in final_results. Keys available: {list(final_results.keys())}")

    rows = []
    priors_template = make_priors_for_K_only(P_days, t0, K_bounds=(0.0, 500.0))

    for Mp in tqdm(Mp_list, desc='Mp loop'):
        dates, RVs_true, K_true = get_RV(P_days, Mp, Ms_solar, N=N, t0=t0)

        mala_RVs, template_RVs, int_RVs, uncs = get_augmented_RVs(RVs_true, final_results, N=N, i=i, snr=snr, nspec=nspec)
        mala_unc, template_unc, int_unc = uncs

        method_dict = {
            'mala': (mala_RVs, mala_unc),
            'template': (template_RVs, template_unc),
            'intrinsic': (int_RVs, int_unc)
        }

        for method_name, (arr_RVs, method_unc) in method_dict.items():
            # n_realizations = arr_RVs.shape[0]
            for realization_idx in range(nreal):
                print(method_name,realization_idx)
                rv_series = arr_RVs[realization_idx, :]
                rv_err_series = np.full_like(rv_series, method_unc)

                # Temporary dir and file
                tmp_dir = tempfile.mkdtemp(prefix='juliet_tmp_')
                tmp_dat = os.path.join(tmp_dir, f'temp_{method_name}_Mp{Mp:.5f}_r{realization_idx}.dat')

                try:
                    write_temp_dat(tmp_dat, dates, rv_series, rv_err_series, instrument_name='HARPS')

                    priors = priors_template.copy()
                    dataset = juliet.load(priors=priors, rvfilename=tmp_dat, out_folder=tmp_dir)
                    results = dataset.fit(n_live_points=n_live_points)

                    K_samples = extract_K_samples(results)
                    K_median = np.median(K_samples)
                    K_minus = K_median - np.percentile(K_samples, 16)
                    K_plus = np.percentile(K_samples, 84) - K_median

                    rows.append({
                        'Mp_jupiter': Mp,
                        'method': method_name,
                        'realization': realization_idx,
                        'K_median': float(K_median),
                        'K_minus': float(K_minus),
                        'K_plus': float(K_plus),
                        'K_true': float(K_true)
                    })
                except Exception as e:
                    print(f"Warning: fit failed for Mp={Mp}, method={method_name}, idx={realization_idx}: {e}")
                    rows.append({
                        'Mp_jupiter': Mp,
                        'method': method_name,
                        'realization': realization_idx,
                        'K_median': np.nan,
                        'K_minus': np.nan,
                        'K_plus': np.nan,
                        'K_true': float(K_true),
                        'error': repr(e)
                    })
                finally:
                    # Clean up juliet output + tmp file
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv}")
    return df

# ---------------------------
# Example usage (run when this file is executed directly)
# ---------------------------
if __name__ == '__main__':
    results_folder = "results"   # change if needed
    df_results = run_full_pipeline(results_folder=results_folder,
                                   pattern_prefix='barnard',
                                   Mp_list=[0.01, 0.03, 0.05, 0.1, 0.5, 1, 5, 10],
                                   P_days=7.0,
                                   Ms_solar=0.12,
                                   t0=2460988.218692,
                                   N=20,
                                   i=0, snr=10, nspec=20,
                                   n_live_points=100,
                                   nreal=50,
                                   out_csv='final_barnards_results_summary_s10.csv')

    df_results = run_full_pipeline(results_folder=results_folder,
                                   pattern_prefix='barnard',
                                   Mp_list=[0.01, 0.03, 0.05, 0.1, 0.5, 1, 5, 10],
                                   P_days=7.0,
                                   Ms_solar=0.12,
                                   t0=2460988.218692,
                                   N=20,
                                   i=0, snr=25, nspec=20,
                                   n_live_points=100,
                                   nreal=50,
                                   out_csv='final_barnards_results_summary_s25.csv')
    
    df_results = run_full_pipeline(results_folder=results_folder,
                                   pattern_prefix='barnard',
                                   Mp_list=[0.01, 0.03, 0.05, 0.1, 0.5, 1, 5, 10],
                                   P_days=7.0,
                                   Ms_solar=0.12,
                                   t0=2460988.218692,
                                   N=20,
                                   i=0, snr=50, nspec=20,
                                   n_live_points=100,
                                   nreal=50,
                                   out_csv='final_barnards_results_summary_s50.csv')
    
    df_results = run_full_pipeline(results_folder=results_folder,
                                   pattern_prefix='barnard',
                                   Mp_list=[0.01, 0.03, 0.05, 0.1, 0.5, 1, 5, 10],
                                   P_days=7.0,
                                   Ms_solar=0.12,
                                   t0=2460988.218692,
                                   N=20,
                                   i=0, snr=75, nspec=20,
                                   n_live_points=100,
                                   nreal=50,
                                   out_csv='final_barnards_results_summary_s75.csv')
    
    # df_results = run_full_pipeline(results_folder=results_folder,
    #                                pattern_prefix='barnard',
    #                                Mp_list=[0.01, 0.03, 0.05, 0.1, 0.5, 1, 5, 10],
    #                                P_days=7.0,
    #                                Ms_solar=0.12,
    #                                t0=2460988.218692,
    #                                N=20,
    #                                i=0, snr=100, nspec=10,
    #                                n_live_points=100,
    #                                nreal=50,
    #                                out_csv='k_results_summary_s100.csv')

    
