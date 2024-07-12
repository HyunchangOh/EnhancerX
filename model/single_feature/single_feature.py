import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

chromosomes = [
        'chr1',
        'chr2',
        'chr3',
        'chr4',
        'chr5',
        'chr6',
        'chr7',
        'chr8',
        'chr9',
        'chr10',
        'chr11',
        'chr12',
        'chr13',
        'chr14',
        'chr15',
        'chr16',
        'chr17',
        'chr18',
        'chr19',
        'chr20',
        'chr21',
        'chr22',
        'chrX'
    ]

# features = ["Interaction","CTCF","cod","DHS","EP300Conservative","h3k4me1","h3k4me2","h3k9me3","h3k27ac","h3k27me3","h3k36me3","promoter_any","promoter_forward","promoter_reverse"]
features = ["h3k27me3","h3k36me3","promoter_any"]

data_path = "../../../../../scratch/ohh98/la_grande_table/"
save_path = "../../../../../scratch/ohh98/models/single_feature/"


for f in features:
    overall_TP = 0
    overall_TN = 0
    overall_FP = 0
    overall_FN = 0
    for c in chromosomes:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        enhancer = np.load(data_path+c+"/"+"enhancer_atlas"+".npy")
        feature = np.load(data_path+c+"/"+f+".npy")
        for i in range(len(enhancer)):
            truth = enhancer[i]
            prediction = feature[i]
            if truth:
                if truth == prediction:
                    TP+=1
                    overall_TP+=1
                else:
                    FN+=1
                    overall_FN+=1
            else:
                if truth == prediction:
                    TN +=1
                    overall_TN+=1
                else:
                    FP +=1
                    overall_FP+=1
        conf_matrix = np.array([[TP, FN], [FP, TN]])
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
        plt.ylabel('Enhancer Atlas')
        plt.xlabel('Naive Predictor: '+f)
        plt.title('Confusion Matrix')
        plt.savefig(save_path+c+"/"+f+'_naive_confusion_matrix.png')
        plt.close()

        fi = open(save_path+c+"/"+f+'_naive_metrics.txt',"w")
        fi.write("Accuracy\tPrecision\tRecall\tF1_score\n")
        accuracy = (TN+TP)/(TN+FP+TP+FN)
        precision = (TP)/(TP+FP)
        recall = (TP)/(TP+FN)
        F1_score = 2*(precision*recall)/(precision+recall)
        fi.write(str(accuracy)+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(F1_score)+"\n")
        fi.close()

    fi = open(save_path+"overall"+"/"+f+'_naive_metrics.txt',"w")
    fi.write("Accuracy\tPrecision\tRecall\tF1_score\n")
    accuracy = (overall_TN+overall_TP)/(overall_TN+overall_FP+overall_TP+overall_FN)
    precision = (overall_TP)/(overall_TP+overall_FP)
    recall = (overall_TP)/(overall_TP+overall_FN)
    F1_score = 2*(precision*recall)/(precision+recall)
    fi.write(str(accuracy)+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(F1_score)+"\n")
    fi.close()

    conf_matrix = np.array([[overall_TP, overall_FN], [overall_FP, overall_TN]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
    plt.ylabel('Enhancer Atlas')
    plt.xlabel('Naive Predictor: '+f)
    plt.title('Confusion Matrix')
    plt.savefig(save_path+"overall"+"/"+f+'_naive_confusion_matrix.png')
    plt.close()



overall_TP = 0
overall_TN = 0
overall_FP = 0
overall_FN = 0
f="ALL_FALSE"
for c in chromosomes:
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    enhancer = np.load(data_path+c+"/"+"enhancer_atlas"+".npy")
    for i in range(len(enhancer)):
        truth = enhancer[i]
        prediction = False
        if truth:
            if truth == prediction:
                TP+=1
                overall_TP+=1
            else:
                FN+=1
                overall_FN+=1
        else:
            if truth == prediction:
                TN +=1
                overall_TN+=1
            else:
                FP +=1
                overall_FP+=1

    conf_matrix = np.array([[TP, FN], [FP, TN]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
    plt.ylabel('Enhancer Atlas')
    plt.xlabel('Naive Predictor: '+f)
    plt.title('Confusion Matrix')
    plt.savefig(save_path+c+"/"+f+'_naive_confusion_matrix.png')
    plt.close()

    fi = open(save_path+c+"/"+f+'_naive_metrics.txt',"w")
    fi.write("Accuracy\tPrecision\tRecall\tF1_score\n")
    accuracy = (TN+TP)/(TN+FP+TP+FN)
    precision = (TP)/(TP+FP)
    recall = (TP)/(TP+FN)
    F1_score = 2*(precision*recall)/(precision+recall)
    fi.write(str(accuracy)+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(F1_score)+"\n")
    fi.close()


fi = open(save_path+"overall"+"/"+f+'_naive_metrics.txt',"w")
fi.write("Accuracy\tPrecision\tRecall\tF1_score\n")
accuracy = (overall_TN+overall_TP)/(overall_TN+overall_FP+overall_TP+overall_FN)
precision = (overall_TP)/(overall_TP+overall_FP)
recall = (overall_TP)/(overall_TP+overall_FN)
F1_score = 2*(precision*recall)/(precision+recall)
fi.write(str(accuracy)+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(F1_score)+"\n")
fi.close()

conf_matrix = np.array([[overall_TP, overall_FN], [overall_FP, overall_TN]])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Positive", "Predicted Negative"], yticklabels=["Actual Positive", "Actual Negative"])
plt.ylabel('Enhancer Atlas')
plt.xlabel('Naive Predictor: '+f)
plt.title('Confusion Matrix')
plt.savefig(save_path+"overall"+"/"+f+'_naive_confusion_matrix.png')
plt.close()

