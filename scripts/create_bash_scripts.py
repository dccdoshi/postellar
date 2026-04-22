import os
'''
This script is meant to create the bashscripts to submit jobs when you want to sample over multiple orders. 
However, this provides the template for the bashscript for one order.


'''

# Folder that contains our trained models. We have trained models per echelle order
base_dir = "../../order_model"  # 

# Get all subfolders inside base_dir to get all the order numbers
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

for folder in subfolders:
    # This gets the folder name for the trained model
    folder_name = os.path.basename(folder.rstrip("/"))  # just the folder name
    order = folder_name[-2:]
    if folder_name.startswith("b"):
        job_name = f"barnards_{order}"
        script_name = f"{order}.sh"
        
        with open(script_name, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-03:00
#SBATCH --account=def-lplevass
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err


source $HOME/pRV/bin/activate
echo "Running job for {order}"
python sampling_with_phoenix_model.py -i 0 -snr 10 25 50 75 -ntemp 10 -order {int(order)} -val "SPIRou{order}_val.df" -output "o{order}" -model "{folder_name}"
""")

        os.chmod(script_name, 0o755)
        print(f"Created {script_name}")
