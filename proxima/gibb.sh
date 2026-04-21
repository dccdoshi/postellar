#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-10:00
#SBATCH --account=def-lplevass
#SBATCH --job-name=mt_o29
#SBATCH --output=./pr_mt_o29%j.out
#SBATCH --error=./pr_mt_o29%j.err
#SBATCH --mail-user="dhvani.doshi@mail.mcgill.ca"
#SBATCH --mail-type=ALL

source $HOME/envs/pRV/bin/activate
echo "Current directory: $(pwd)"
python gibbs_experiment.py -i 0 -snr 50 75 100 -ntemp 10 -order 29 -val SPIRou29_val.df -output mt_o29 -model b8nf16ch2_2_2_2_e750_o29