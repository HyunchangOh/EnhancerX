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
|   ├── hg19/ *reference
|   |   └── chr1.fa
|   ├── hg38/ *reference
|   |   └── chr1.fa
|   ├── VISTA/ *preprocessed!
|   |   └── vista.txt
|   ├── EPD/
|   |   └── Hs_EPDnew_006_hg19.bed
|   └── enhancer_atlas/ *preprocessed!
|       └── GM12878.txt
├── preprocess/
|   ├── GC_content/
|   |   └── enhancer_atlas_AND_hg19.py
|   └── read_gen_db.py
├── processed/
|   ├── VISTA/
|   |   └── vista.tsv 
|   ├── la_grande_table.tsv
|   ├── enrichment_GC_PER_sequence.tsv
|   └── la_grande_table_gen.py
├── model/
|   ├── proto_HMM.py
|   ├── proto_CNN.py
|   └── prediction.txt
├── .gitignore
└── README.md
```

Goal is to make this table as large as possible WITHOUT HARMING THE INTEGRITY

* enrichment_GC_PER_sequence.tsv
sequence(hg19 chr1) / GC_content / enrichment_score (enhancerAtlas GM12878 chr1)

## Data 
Put all raw data into 'data' directory. Make a subdirectory for each database you use.

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
Generate la grande table from la_grande_table_gen.py, in folder processed/.  
For now, it takes data from hg19, and Enhancer Atlas.  

## Models
Different models here.

### HMM
Currently in a proto state.  
Run directly.  
Current relevant values for analysis right now are: 
- Chromosome 1.
- Positions 0 through 2M.
- 2-state-markov chain (non-enhancer or enhancer, by base pair).

### CNN
Currently in proto state.  
Run directly.  
- Definable chromosome number.
- Definable base pair range.
- 800 wide input layer (sliding window).

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