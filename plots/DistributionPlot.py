import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

# Load the data from the CSV file
df = pd.read_csv('Cleaned_Merged_Ratio_Results.csv')

# Create a list of unique chromosomes and features
chromosomes = df['Chromosome'].unique()
features = df['Feature'].unique()

# 1. Heatmap of Feature Correlations Across Chromosomes
pivot_df = df.pivot_table(index='Chromosome', columns='Feature', values='Ratio')
correlation_matrix = pivot_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Heatmap of Feature Correlations Across Chromosomes')
plt.tight_layout()
plt.savefig('heatmap_feature_correlations.png')
plt.close()

# 2. Scatter Plot for the Distribution of Ratios per Feature for Each Chromosome
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

# 3. Line Plot to Observe Trends in Ratios Across Chromosomes for Each Feature
plt.figure(figsize=(14, 8))
for feature in features:
    feature_data = df[df['Feature'] == feature].groupby('Chromosome')['Ratio'].mean().reset_index()
    plt.plot(feature_data['Chromosome'], feature_data['Ratio'], marker='o', label=feature)
plt.title('Line Plot of Average Ratios per Chromosome by Feature')
plt.xlabel('Chromosome')
plt.ylabel('Average Ratio')
plt.xticks(rotation=45)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('lineplot_avg_ratios_per_chromosome.png')
plt.close()

# 4. Bar Plot to Compare the Average Ratio for Each Chromosome
plt.figure(figsize=(14, 8))
avg_ratios = df.groupby('Chromosome')['Ratio'].mean().reset_index()
sns.barplot(x='Chromosome', y='Ratio', data=avg_ratios)
plt.title('Bar Plot of Average Ratios Across Chromosomes')
plt.xlabel('Chromosome')
plt.ylabel('Average Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('barplot_avg_ratios_across_chromosomes.png')
plt.close()

# 5. Strip Plot to Visualize Individual Data Points for Each Chromosome
plt.figure(figsize=(14, 8))
sns.stripplot(x='Chromosome', y='Ratio', data=df, jitter=True)
plt.title('Strip Plot of Ratios Across Chromosomes')
plt.xlabel('Chromosome')
plt.ylabel('Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('stripplot_ratios_across_chromosomes.png')
plt.close()

# 6. Bar Plot of Ratios by Chromosome and Feature
plt.figure(figsize=(14, 8))
sns.barplot(x='Chromosome', y='Ratio', hue='Feature', data=df)
plt.title('Bar Plot of Ratios by Chromosome and Feature')
plt.xlabel('Chromosome')
plt.ylabel('Ratio (Zeros to Non-Zeros)')
plt.xticks(rotation=45)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('barplot_ratios_by_chromosome_and_feature.png')
plt.close()

# 7. Stacked Bar Plot of Zeros and Non-Zeros by Chromosome and Feature
df['Total Elements'] = df['Number of zeros'] + df['Number of non-zero elements']
df_melted = df.melt(id_vars=['Chromosome', 'Feature'], value_vars=['Number of zeros', 'Number of non-zero elements'], var_name='Element Type', value_name='Count')

plt.figure(figsize=(14, 8))
sns.barplot(x='Chromosome', y='Count', hue='Element Type', data=df_melted, ci=None, palette='muted')
plt.title('Stacked Bar Plot of Zeros and Non-Zeros by Chromosome and Feature')
plt.xlabel('Chromosome')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Element Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('stacked_barplot_zeros_nonzeros_by_chromosome.png')
plt.close()

# 8. Scatter Plot of Ratios with Size Indicating Total Counts
plt.figure(figsize=(14, 8))
sizes = df['Total Elements'] / df['Total Elements'].max() * 1000  # Normalize sizes for plotting
sns.scatterplot(x='Chromosome', y='Ratio', size=sizes, hue='Feature', data=df, sizes=(20, 200))
plt.title('Scatter Plot of Ratios with Size Indicating Total Counts')
plt.xlabel('Chromosome')
plt.ylabel('Ratio (Zeros to Non-Zeros)')
plt.xticks(rotation=45)
plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('scatterplot_ratios_with_sizes.png')
plt.close()

# 9. Facet Grid for Attribute Ratios per Chromosome
g = sns.FacetGrid(df, col="Chromosome", col_wrap=4, height=4, margin_titles=True)
g.map(sns.barplot, "Feature", "Ratio")
g.set_axis_labels("Feature", "Ratio (Zeros to Non-Zeros)")
g.set_titles(col_template="Chromosome {col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Bar Plot of Ratios per Feature by Chromosome', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
g.savefig('facetgrid_ratios_per_chromosome.png')
plt.close()

# Statistical Tests for Associations

# Example 1: Chi-square test for association between Feature and Chromosome
chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Feature'], df['Chromosome']))
results = pd.DataFrame({'Chi-square': [chi2], 'p-value': [p]})
results.to_csv('chi2_feature_chromosome.csv', index=False)

# Example 2: Cramer's V for association between Feature and Chromosome
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

cramers_v_score = cramers_v(df['Feature'], df['Chromosome'])
cramers_v_result = pd.DataFrame({'Cramer\'s V': [cramers_v_score]})
cramers_v_result.to_csv('cramers_v_feature_chromosome.csv', index=False)

# Example 3: Chi-square test for association between Ratio Category (as a categorical variable) and Chromosome
df['Ratio Category'] = pd.qcut(df['Ratio'], q=3, labels=['Low', 'Medium', 'High'])
chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Ratio Category'], df['Chromosome']))
results = pd.DataFrame({'Chi-square': [chi2], 'p-value': [p]})
results.to_csv('chi2_ratio_category_chromosome.csv', index=False)

# Example 4: Chi-square test for association between Number of 0s and Chromosome
chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Number of zeros'], df['Chromosome']))
results = pd.DataFrame({'Chi-square': [chi2], 'p-value': [p]})
results.to_csv('chi2_number_of_zeros_chromosome.csv', index=False)

# Example 5: Chi-square test for association between Number of non-0s and Chromosome
chi2, p, _, _ = chi2_contingency(pd.crosstab(df['Number of non-zero elements'], df['Chromosome']))
results = pd.DataFrame({'Chi-square': [chi2], 'p-value': [p]})
results.to_csv('chi2_number_of_nonzeros_chromosome.csv', index=False)