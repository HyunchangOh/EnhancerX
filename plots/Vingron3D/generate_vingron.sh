#!/bin/bash
#SBATCH --job-name=generate_vingron
#SBATCH --output=output_%j.txt  # Save output to a file named output_<job_id>.txt
#SBATCH --error=error_%j.txt    # Save error messages to a file named error_<job_id>.txt
#SBATCH --time=04:00:00         # Set a maximum time limit for the job (1 hour in this case)
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Request 1 task (process)
#SBATCH --cpus-per-task=1       # Request 1 CPU per task
#SBATCH --qos=standard          # Specify the Quality of Service
#SBATCH --mem=16G                # Specify the total memory required for the job (example: 4GB)
#SBATCH --mail-user=ohh98@zedat.fu-berlin.de


export LMOD_DISABLE_SAME_NAME_AUTOSWAP="no"
module add SciPy-bundle
module add matplotlib

# Run the Python script
python generate_vingron.py

#18239798 (1-4 job number)