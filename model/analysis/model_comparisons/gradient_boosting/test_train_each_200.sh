#!/bin/bash
#SBATCH --job-name=GB_LR2
#SBATCH --output=output200_each_train_test_%j.txt  # Save output to a file named output_<job_id>.txt
#SBATCH --error=erro200_each_train_testr_%j.txt    # Save error messages to a file named error_<job_id>.txt
#SBATCH --time=24:00:00         # Set a maximum time limit for the job (1 hour in this case)
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Request 1 task (process)
#SBATCH --cpus-per-task=1       # Request 1 CPU per task
#SBATCH --qos=standard          # Specify the Quality of Service
#SBATCH --mem=64G                # Specify the total memory required for the job (example: 4GB)
#SBATCH --mail-user=ohh98@zedat.fu-berlin.de
export LMOD_DISABLE_SAME_NAME_AUTOSWAP="no"
module add SciPy-bundle/2021.05-foss-2021a
module add matplotlib/3.4.2-foss-2021a
module add TensorFlow/2.6.0-foss-2021a
module add scikit-learn/0.24.2-foss-2021a
# module add Seaborn


# Run the Python script
python test_train_each_200.py

# a_better_forest -> 18254598
