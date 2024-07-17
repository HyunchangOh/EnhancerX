# EnhancerX
Annotate enhancer regions from other genomic annotations using various machine-learning techniques. 


### Data
We are dealing with big data, therefore we do not upload it directly to the GitHub repo. The data can be access through link: 


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

## Data 

The following explains how the data was accessed and what they are.

### VISTA
| **Field**           | **Example**     | **Description**                                          |
|---------------------|-----------------|----------------------------------------------------------|
| **chrom**           | chr1            | Chromosome (or contig, scaffold, etc.)                   |
| **chromStart**      | 167327716       | Start position in chromosome                             |
| **chromEnd**        | 167329809       | End position in chromosome                               |
| **name**            | hs1331          | Name of item                                             |
| **score**           | 0               | Score from 0-1000                                        |
| **strand**          | .               | + or -                                                   |
| **thickStart**      | 167327717       | Start of where display should be thick (start codon)     |
| **thickEnd**        | 167329809       | End of where display should be thick (stop codon)        |
| **color**           | 255,0,0         | Primary RGB color for the decoration                     |
| **patternExpression**| positive       | Observed spatial pattern of expression                   |
| **experimentId**    | 1331            | Experiment ID    

https://enhancer.lbl.gov/cgi-bin/imagedb3.pl?page_size=10000;search.form=no;page=1;action=search;search.result=yes;show=1;form=search;search.sequence=1

Coordinates reference hg19 (human) mm9 (mouse)

### hg19
Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/  
Scrolling down to find the link. You do not need FTP for this.

### hg38
Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/  
Source: https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/  
The link above leads to big zip files that contain all the data.  
The link below leads to separate files for each chromosome.

### EPD
Source: https://epd.expasy.org/ftp/epdnew/ 
head:
chromosome  start   end     name    score   strand  thick(transcribed start)    thick(transcribed end)
chr1	    894625	894685	NOC2L_1	900	    -	    894625	                    894636

preprocessed:
annotate 0/1 for forward-promoters
annotate 0/1 for reverse-promoters
annotate 0/1 for thick-forward-promoters
annotate 0/1 for thick-transcribed-promoters

### enhanceratlas.org
General source: http://enhanceratlas.org/downloadv2.php

Files are in BED (text) format, with four (4) columns in each file (at "Download enhancers"):  
chrom-Enh - Name of the chromosome for enhancer;  
chromStart - The starting position of enhancer;  
chromEnd - The ending position of enhancer;  
enhancer signal - enrichment score.  

Simply download cell line/tissue file using whatever means.  
If you are using windows you will probably want to use $Invoke-WebRequest -Uri "enhancer_file_URL", or maybe even ctrl+a, copy to .txt file (files are not too large).    
If you are using linux, a simple $wget.  

Ideally would download "Enhancers of all species by bed format", but files are very large, species -> cell line/tissue will have to do for now.  

Coordinates reference hg19

### La grande table
La grande table is a structure of folders as defined in the folder structure above. Every chromosome folder contains the npy files required to load all features (one npy file per feature), each one being a column in the table.  
It is automatically created running processed/la_grande_table_gen.py, using whichever fasta files there are loaded in data/hg19/ for the sequence ("seq") and coding/non-coding annotation ("cod"), and files in data/enhancer_atlas/ for enhancer/non-enhancer annotation ("atl").  
Function mucho_load is defined, which takes chromosome number (string format: "chr1", "chr2", etc) and list of features or columns to load (list format: ["seq", "cod", "atl"]) as input paramenters, and returns la grande table as a numpy array with shape (no_cols, length_seq). To access a specific data point, for example: la_grande_table[seq][100234].  


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
