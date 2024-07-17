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


