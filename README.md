# Enhancer X

Predicting enhancers is challenging due to several factors: they are located in vast non-coding regions of the genome, lack consistent sequence motifs, and their activity varies across different cell types, developmental stages, and environmental conditions. Additionally, enhancers often exhibit redundancy and can function modularly, further complicating their identification. Accurate prediction of enhancers is crucial because they are involved in many biological processes and diseases. 

Our goal with Enhancer X is to predict enhancers using different genomic annotations. By combining various data sources and using advanced computational methods, Enhancer X aims to improve the accuracy and reliability of enhancer prediction. This, in turn, will enhance our understanding of gene regulation and its implications for health and disease. Specifically, we developed two models: one for predicting enhancers based on interactions and another for predicting enhancers from regions. 

This GitHub repository contains all the code used for creating the region model, including data preprocessing, various model constructions and data visualizations. Since we are dealing with big data, it will not be uploaded to the GitHub repo. The sub-sampled data can be accessed through the following link:  
https://drive.google.com/file/d/1xoGo9fNkEYFD-tSqfSNv6_HMYIbnYulb/view?usp=sharing

Data is structured as one folder for each chromosome, and inside, one .npy file per feature (already binned and sub-sampled). As an example, the file that stores the 3D distances from h3k4me1 methylation sites to promoters, from chromosome 1, is located at:  
/Susampled_Final/chr1/BIN50_h3k4me1_3D_Dist.npy  
See list of features below.


## Workflow
![Enhancer X Workflow Overview](OverviewWorkflow.png) 

## Orgnisation
'''
EnhancerX/
|
├── data/*omitted on Github
|   ├── hg19/ *reference
|   |   └── chr1.fa
|   ├── ...
|   └── enhancer_atlas/
|       └── GM12878.txt
|
├── la_grande_table/*omitted on Github
|   ├── chr1/
|   ├── chr2/
|   ├── ...
|   └── chrY/
|
├── preprocess/
|   ├── 3DIV/
|   ├── ENCODE/
|   |   ├── CTCF/
|   |   ├── .../ *more directories per feature
|   |   └── h3k36me3/
|   |        ├── annotate.py *reads raw data to create boolean annotation
|   |        ├── calculate_1D_distance.py *calculate 1D distance to nearest feature
|   |        └── calculate_3D_distance.py *calculate 3D distance to nearest feature
|   |
|   ├── .../ *more directories per database
|   ├── SubSampling/ *performs subsampling
|   └── bin/ *performs binning
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
'''
### Preprocess

## Features  

Tip: file_name = "BIN50_" + feature_name + ".npy"  

### Predictor variable  

  enhancer_atlas


### List of features  

  cod  
  cod_1D_Dist  

  promoter_any  
  promoter_1D_Dist_any  
  promoter_3D_Dist_any  
  promoter_forward  
  promoter_1D_Dist_forward  
  promoter_reverse  
  promoter_1D_Dist_reverse  

  h3k4me1  
  Bh3k4me1_1D_Dist  
  Bh3k4me1_3D_Dist  

  h3k4me2  
  Bh3k4me2_1D_Dist  
  h3k4me2_3D_Dist  

  h3k9me3  
  h3k9me3_1D_Dist  
  h3k9me3_3D_Dist  

  h3k27ac  
  h3k27ac_1D_Dist  
  h3k27ac_3D_Dist  

  h3k27me3  
  h3k27me3_1D_Dist  
  h3k27me3_3D_Dist  

  h3k36me3  
  h3k36me3_1D_Dist  
  h3k36me3_3D_Dist  

  CTCF  
  CTCF_1D_Dist  
  CTCF_3D_Dist  
    
  DHS  
  DHS_1D_Dist  
  DHS_3D_Dist  

  EP300Conservative  
  EP300Conservative_1D_Dist  
  EP300Conservative_3D_Dist  

  Interaction_1D_Dist  
