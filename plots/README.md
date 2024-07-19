# Plotting Pipeline Overview

## Steps

### Ratio Calculation 
 The script `RatioCalculation.py` loads data from the `la_grande_table` and performs 1D calculations. It processes the data to categorize values: numbers as possible sites (non-zero) and zeros as not. It then computes the ratio of possible sites to total entries, representing their density along chromatin. The results are saved to `Cleaned_Merged_Ratio_Results.csv`.

### Distribution Plots 
The script performs data visualization for statistical analysis of distances. It takes the main table generated from `Cleaned_Merged_Ratio_Results.csv` and creates the following plots:

 1. **Heatmap of Feature Correlations Across Chromosomes:** This plot shows the correlation matrix between different features across chromosomes using a heatmap.
 2. **Scatter Plot of Ratios per Feature by Chromosome:** This plot displays scatter plots for each feature, showing the distribution of ratios across different chromosomes.
 3. **Line Plot of Average Ratios per Chromosome by Feature:** This plot illustrates the average ratio trend across chromosomes for each feature using line plots.
 4. **Bar Plot of Average Ratios Across Chromosomes:** This plot provides a bar plot showing the average ratio across all chromosomes.
 5. **Strip Plot of Ratios Across Chromosomes:** This plot shows individual data points (ratios) across chromosomes using a strip plot.
 6. **Facet Grid of Histograms:** This facet grid shows histograms of ratios for each feature across different chromosomes.
 7. **Bar Plot of Ratios by Chromosome and Feature:** This plot displays a bar plot that compares ratios across both chromosomes and features.

## Packages
| Package               | Import Statement               | Purpose                                                                                   | Functions                                                                                       |
|-----------------------|--------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| numpy                 | `import numpy as np`           | Numerical computing and array operations                                                   | Efficient storage and manipulation of numerical arrays                                          |
| pandas                | `import pandas as pd`          | Data manipulation and analysis                                                             | Reading, writing, and transforming data structures like DataFrames                               |
| matplotlib.pyplot     | `import matplotlib.pyplot as plt` | Creating static, animated, and interactive visualizations                                | MATLAB-like plotting interface for various plot types like line plots, scatter plots, histograms |
| seaborn               | `import seaborn as sns`        | Enhancing visual appeal and adding statistical functionalities to matplotlib plots         | High-level functions for creating informative statistical graphics                               |
