# Plotting Pipeline Overview

## Steps

### Ratio Calculation 
 The script `RatioCalculation.py` loads data from the `la_grande` table and performs 1D calculations. It processes the data to categorize values: numbers as possible sites (non-zero) and zeros as not. It then computes the ratio of possible sites to total entries, representing their density along chromatin. The results are saved to `Cleaned_Merged_Ratio_Results.csv`.

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
| glob                  | `import glob`                  | File path expansion                                                                        | File pattern matching and retrieval of file names or paths based on patterns                     |
| os                    | `import os`                    | Operating system functionalities                                                           | File manipulation, directory operations, and interaction with the operating system               |
| matplotlib.pyplot     | `import matplotlib.pyplot as plt` | Creating static, animated, and interactive visualizations                                | MATLAB-like plotting interface for various plot types like line plots, scatter plots, histograms |
| seaborn               | `import seaborn as sns`        | Enhancing visual appeal and adding statistical functionalities to matplotlib plots         | High-level functions for creating informative statistical graphics                               |


### The following have been reloaded with a version change:
  1) FFTW/3.3.10-GCC-12.3.0 => FFTW/3.3.8-gompi-2020b
  2) GCC/12.3.0 => GCC/10.2.0
  3) GCCcore/12.3.0 => GCCcore/10.2.0
  4) LibTIFF/4.5.0-GCCcore-12.3.0 => LibTIFF/4.1.0-GCCcore-10.2.0
  5) NASM/2.16.01-GCCcore-12.3.0 => NASM/2.15.05-GCCcore-10.2.0
  6) OpenBLAS/0.3.23-GCC-12.3.0 => OpenBLAS/0.3.12-GCC-10.2.0
  7) Pillow/10.0.0-GCCcore-12.3.0 => Pillow/8.0.1-GCCcore-10.2.0
  8) Python/3.11.3-GCCcore-12.3.0 => Python/3.8.6-GCCcore-10.2.0
  9) SQLite/3.42.0-GCCcore-12.3.0 => SQLite/3.33.0-GCCcore-10.2.0
 10) SciPy-bundle/2023.07-gfbf-2023a => SciPy-bundle/2020.11-foss-2020b
 11) Tcl/8.6.13-GCCcore-12.3.0 => Tcl/8.6.10-GCCcore-10.2.0
 12) Tk/8.6.13-GCCcore-12.3.0 => Tk/8.6.10-GCCcore-10.2.0
 13) Tkinter/3.11.3-GCCcore-12.3.0 => Tkinter/3.8.6-GCCcore-10.2.0
 14) X11/20230603-GCCcore-12.3.0 => X11/20201008-GCCcore-10.2.0
 15) XZ/5.4.2-GCCcore-12.3.0 => XZ/5.2.5-GCCcore-10.2.0
 16) binutils/2.40-GCCcore-12.3.0 => binutils/2.35-GCCcore-10.2.0
 17) bzip2/1.0.8-GCCcore-12.3.0 => bzip2/1.0.8-GCCcore-10.2.0
 18) expat/2.5.0-GCCcore-12.3.0 => expat/2.2.9-GCCcore-10.2.0
 19) fontconfig/2.14.2-GCCcore-12.3.0 => fontconfig/2.13.92-GCCcore-10.2.0
 20) freetype/2.13.0-GCCcore-12.3.0 => freetype/2.10.3-GCCcore-10.2.0
 21) libffi/3.4.4-GCCcore-12.3.0 => libffi/3.3-GCCcore-10.2.0
 22) libjpeg-turbo/2.1.5.1-GCCcore-12.3.0 => libjpeg-turbo/2.0.5-GCCcore-10.2.0
 23) libpciaccess/0.17-GCCcore-12.3.0 => libpciaccess/0.16-GCCcore-10.2.0
 24) libpng/1.6.39-GCCcore-12.3.0 => libpng/1.6.37-GCCcore-10.2.0
 25) libreadline/8.2-GCCcore-12.3.0 => libreadline/8.0-GCCcore-10.2.0
 26) matplotlib/3.7.2-gfbf-2023a => matplotlib/3.3.3-foss-2020b
 27) ncurses/6.4-GCCcore-12.3.0 => ncurses/6.2-GCCcore-10.2.0
 28) pybind11/2.11.1-GCCcore-12.3.0 => pybind11/2.6.0-GCCcore-10.2.0
 29) util-linux/2.39-GCCcore-12.3.0 => util-linux/2.36-GCCcore-10.2.0
 30) xorg-macros/1.20.0-GCCcore-12.3.0 => xorg-macros/1.19.2-GCCcore-10.2.0
 31) zlib/1.2.13-GCCcore-12.3.0 => zlib/1.2.11-GCCcore-10.2.0
