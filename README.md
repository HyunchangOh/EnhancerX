# EnhancerX
Annotate Enhancers using various ML techniques

## INTERN
DO NOT upload data to github repo directly. 

The folder 'data' is added to .gitignore, so anything inside that folder will be ignored by git.

If you write a code, or modify anything that uses a new data, update README.md - Data section and Organization section on how to download/unzip the raw data accordingly.

Provide whatever necessary tech-tutorials to "TroubleShooting"

Use tsv, not csv if possible. 

READ THE ORGANIZATION AND STICK TO THE STRUCTURE
BE ORGANIZED

## Organization
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
|   └── enrichment_GC_PER_sequence.tsv
├── model/
├── .gitignore
└── README.md
```

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
This wii REMOVE the gz file and create a new file named chr1.fa