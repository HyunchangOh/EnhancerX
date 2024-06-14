# EnhancerX
Annotate Enhancers using various ML techniques

## INTERN_Rules

READ THE ORGANIZATION AND STICK TO THE STRUCTURE  
BE ORGANIZED  
Rules can be changed through democratic means.

### Data
DO NOT upload data to github repo directly. 

The folder 'data' is added to .gitignore, so anything inside that folder will be ignored by git.

### Log
If you write a code, or modify anything that uses a new data, update README.md - Data section and Organization section on how to download/unzip the raw data accordingly.

Provide whatever necessary tech-tutorials to "TroubleShooting"

Writing small logs on DEVDIARY.md may help others and help you get back to work quickly later.

### Formats
Use tsv, not csv if possible. (decimal point , . conversion error can screw things up)
Use hard typing if possible.
Use docstring if possible.
Showing Database Format (header + head(1)) in a comment can help, whereever you call data from a database.


## Organization
Minor helper functions are skipped
```
EnhancerX/
├── data/
|   ├── hg19/
|   |   └── chr1.fa
|   ├── hg38/
|   |   └── chr1.fa
|   └── enhancer_atlas/
|       └── GM12878.txt
├── preprocess/
|   └── GC_content/
|       └── enhancer_atlas_AND_hg19.py
├── processed/
|   ├── la_grande_table.tsv
|   └── enrichment_GC_PER_sequence.tsv
├── model/
├── .gitignore
└── README.md
```
la_grande_table.tsv is the super large table with all the features/responses/pseudo-responses/ whatever information that may be helpful for us.

Goal is to make this table as large as possible WITHOUT HARMING THE INTEGRITY

* enrichment_GC_PER_sequence.tsv
sequence(hg19 chr1) / GC_content / enrichment_score (enhancerAtlas GM12878 chr1)

## Data 
Put all raw data into 'data' directory. Make a subdirectory for each database you use.

The following explains how the data was accessed and what they are.

### hg19
Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/  
Scrolling down to find the link. You do not need FTP for this.

### hg38
Source: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/  
Source: https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/  
The link above leads to big zip files that contain all the data.  
The link below leads to separate files for each chromosome.

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