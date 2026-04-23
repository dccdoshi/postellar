#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-03:00
#SBATCH --account=def-lplevass
#SBATCH --job-name=test2
#SBATCH --output=test2_%j.out
#SBATCH --error=test2_%j.err


source $HOME/pRV/bin/activate
echo "Running job for {order}"
python sampling_with_phoenix_model.py -i 0 -snr 50 -ntemp 10 -order 20 -val "SPIRou20_val.df" -output "2o20" -model "b8nf16ch2_2_2_2_e500_o20"