import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('Cleaned_Merged_Ratio_Results.csv')

# Create a list of unique chromosomes
chromosomes = df['Chromosome'].unique()
features = df['Feature'].unique()

# Pivot the table to have features as columns and chromosomes as rows, with the ratio as values
pivot_df = df.pivot_table(index='Chromosome', columns='Feature', values='Ratio')

# Compute the correlation matrix
correlation_matrix = pivot_df.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Correlation Coefficient')
plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(correlation_matrix.index)), labels=correlation_matrix.index)
plt.title('Heatmap of Feature Correlations Across Chromosomes')
plt.tight_layout()
plt.savefig('heatmap_feature_correlations.png')
plt.close()
 
# Scatter plot for the distribution of ratios per feature for each chromosome
plt.figure(figsize=(14, 8))
for feature in features:
    feature_data = df[df['Feature'] == feature]
    plt.scatter(feature_data['Chromosome'], feature_data['Ratio'], label=feature)
plt.title('Scatter Plot of Ratios per Feature by Chromosome')
plt.xlabel('Chromosome')
plt.ylabel('Ratio')
plt.xticks(rotation=45)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('scatterplot_ratios_per_feature.png')
plt.close()
