#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-20:00
#SBATCH --account=def-lplevass
#SBATCH --job-name=o04
#SBATCH --output=./o04%j.out
#SBATCH --error=./o04%j.err
#SBATCH --mail-user="dhvani.doshi@mail.mcgill.ca"
#SBATCH --mail-type=ALL

source $HOME/envs/pRV/bin/activate
echo "Current directory: $(pwd)"
python gibbs_experiment.py -i 41 223 -snr 50 100 -ntemp 10 -order 4  -val SPIRou04_val.df -output o04 -model b8nf16ch2_2_2_2_e750_o04