import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# f1
data = {
    'Logistic Regression': [0.67, 0.67, 0.67, 0.67, 0.67],
    'MLP': [0, 0, 0.67, 0, 0.67],
    'Gradient Boosting': [1.00, 1.00, 1.00, 1.00, 1.00],
    'Random Forest': [1.00, 1.00, 1.00, 1.00, 1.00]
}

# #roc auc
# data = {
#     'Logistic Regression': [0.57, 0.57, 0.58, 0.57, 0.57],
#     'MLP': [0.5, 0.5, 0.5, 0.5, 0.5],
#     'Gradient Boosting': [1.00, 1.00, 1.00, 1.00, 1.00],
#     'Random Forest': [1.00, 1.00, 1.00, 1.00, 1.00]
# }

#accuracy
# data = {
#     'Logistic Regression': [0.57, 0.56, 0.56, 0.57, 0.57],
#     'MLP': [0.5, 0.5, 0.5, 0.5, 0.5],
#     'Gradient Boosting': [1.00, 1.00, 1.00, 1.00, 1.00],
#     'Random Forest': [1.00, 1.00, 1.00, 1.00, 1.00]
# }

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate the mean and standard deviation for each model
mean_scores = df.mean()
std_scores = df.std()

# Create the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_scores.index, y=mean_scores.values, yerr=std_scores.values, capsize=0.1)
plt.title('F1 Score')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.savefig("results_f1.png")
plt.show()
