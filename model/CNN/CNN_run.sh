#!/bin/bash
#SBATCH --job-name=run_NN
#SBATCH --output=output_%j.txt  # Save output to a file named output_<job_id>.txt
#SBATCH --error=error_%j.txt    # Save error messages to a file named error_<job_id>.txt
#SBATCH --time=36:00:00         # Set a maximum time limit for the job (1 hour in this case)
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Request 1 task (process)
#SBATCH --cpus-per-task=1       # Request 1 CPU per task
#SBATCH --qos=standard          # Specify the Quality of Service
#SBATCH --mem=64G              # Specify the total memory required for the job (example: 4GB)
#SBATCH --mail-user=ohh98@zedat.fu-berlin.de
export LMOD_DISABLE_SAME_NAME_AUTOSWAP="no"

# Load necessary modules
module add SciPy-bundle/2021.05-foss-2021a
#module add matplotlib/3.4.2-foss-2021a
module add scikit-learn/0.24.2-foss-2021a
module add TensorFlow/2.6.0-foss-2021a

# Run the Python script
python CNN.py

# 18281477 same thing on "final" (finally(?)) data folder """""""""""""""""ENLIGHTMENT""""""""""""""""" looks like: higher batch size, lower learning rate
# 18281529 running for 2 different sets of features: distances, and features themselves """"""""""""""ran"""""""""""""" looks like: same. also, 
# 18281875 runnnig some new hyperparameters based on enlightment, and 3 different feature sets """""""""run""""""""" a bit wierd?
# 18282052 probably last try before heavy all chr run
# 18282143 for fun

# RUNNING:
# 18284991 quick run on chr21 to show hyperparameters and feature lists at presentation """""" no precision division 0?
# 18284992 same thing as last one but chr1

# 18285000 run hyperparamenters on chromosome number all/

