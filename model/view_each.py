import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'a' is your 2D numpy array

features = ["f1","accuracy","precision","recall","roc_auc"]
# models = ["gradient_boosting","logistic_regression","HANNs","random_forest"]
models=["random_forest"]
for m in models:
    for f in features:
        a=np.load('/scratch/ohh98/models/'+m+'/each_chromosome_subsampled/'+f+'.npy')
        labels = [str(i) for i in range(1, a.shape[0])] + ['X']
        # Create a heatmap
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(a, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis",xticklabels=labels, yticklabels=labels)

        # Display the heatmap
        plt.savefig("new_plots_each/"+f+"/"+m+".png")
