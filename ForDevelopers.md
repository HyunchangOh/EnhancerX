# Developers Handbook
This is a practical handbook for developers, an exthensive documentation of the organization files, the troubleshooting, along with HPC commands.

## Organization
Minor helper functions are skipped
```
EnhancerX/
|
├── data/
|   ├── hg19/ *reference
|   |   └── chr1.fa
|   ├── hg38/ *reference
|   |   └── chr1.fa
|   ├── VISTA/ *preprocessed!
|   |   └── vista.txt
|   ├── EPD/
|   |   └── Hs_EPDnew_006_hg19.bed
|   ├── ENCODE/
|   |   └── ENCFF795IDC.bed #GM12878 cell line, h3k27ac, hg19
|   └── enhancer_atlas/ *preprocessed!
|       └── GM12878.txt
|
├── la_grande_table/
|   ├── chr1/
|   ├── chr2/
|   ├── ...
|   └── chrY/
|
├── preprocess/
|   └── EPD/
|       └── annotate.py # reads Hs_EPDnew_006_hg19.bed / writes in la_grande_table
|
├── processed/
|   ├── VISTA/
|   |   └── vista.tsv 
|   ├── enrichment_GC_PER_sequence.tsv
|   └── la_grande_table_gen.py
|
├── model/
|   ├── Random Forest/
|   ├── MLP/
|   ├── logistic_regression
|   └── gradient_boosting
|
├── plots/
|   ├── Vingron
|   └── Vingron3D
|
├── .gitignore
└── README.md
```

* enrichment_GC_PER_sequence.tsv
sequence(hg19 chr1) / GC_content / enrichment_score (enhancerAtlas GM12878 chr1)


## TroubleShooting
Here are some instructions to tackle frequently encountered problems or tips for technical issues.

### 1. How to Unzip .gz File in Windows
Tested System: Windows10 Version 22H2.
Tested Date: 13.06.2024
1. Download gzip here: https://gnuwin32.sourceforge.net/packages/gzip.htm
Download - Complete package, except sources - Setup (Last Change 15 October 2007)

2. Add gzip to PATH
Default path is C:\Program Files (x86)\GnuWin32\bin
You can check the path to gzip.exe in the installer while installing gzip.

3. Open Terminal and navigate to the folder where your .gz file is.

4. gzip -d chr1.fa.gz
This will REMOVE the gz file and create a new file named chr1.fa

### 2. How to Use HPC
Never Run Demanding Job on Login Node, instead use job manager (slurm).
1. First, make a .sh file similar to this.
```
#!/bin/bash
#SBATCH --job-name=ohh98_calc_dist
#SBATCH --output=output_%j.txt  # Save output to a file named output_<job_id>.txt
#SBATCH --error=error_%j.txt    # Save error messages to a file named error_<job_id>.txt
#SBATCH --time=04:00:00         # Set a maximum time limit for the job (1 hour in this case)
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Request 1 task (process)
#SBATCH --cpus-per-task=1       # Request 1 CPU per task
#SBATCH --qos=standard          # Specify the Quality of Service
#SBATCH --mem=16G                # Specify the total memory required for the job (example: 4GB)
#SBATCH --mail-user=ohh98@zedat.fu-berlin.de

# Run the Python script
python calculate_1D_distance.py
```
2. Use the following commands to install modules. (example: numpy)
```
export LMOD_DISABLE_SAME_NAME_AUTOSWAP="no"
module add SciPy-bundle
```
3. Send and Check
```
sbatch your_file_name.sh        #send the file. remember the job id
scontrol show job 18212601      #check the job status (shows also when the job is pending)
squeue --me                     #check all jobs submitted by me (only those that are running!)
```
4. OOM Error
If OOM error happens, increase the memory by changing the .sh file. Give some larger number
```
#SBATCH --mem=8G
```
