#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-20:00
#SBATCH --account=def-lplevass
#SBATCH --job-name=ntemp
#SBATCH --output=./ntemp%j.out
#SBATCH --error=./ntemp%j.err
#SBATCH --mail-user="dhvani.doshi@mail.mcgill.ca"
#SBATCH --mail-type=ALL

source $HOME/pRV/bin/activate
echo "Current directory: $(pwd)"
python gibbs_experiment.py -i 0 -snr 50 -ntemp 10 20 30 40 50 60 70 -order 4  -val SPIRou04_val.df -output seededntemp -model b8nf16ch2_2_2_2_e750_o04