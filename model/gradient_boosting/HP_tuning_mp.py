import numpy as np
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle
from scipy import stats
import scipy

# Define paths
input_dir = '../../../../../scratch/ohh98/la_grande_table_subsampled/subsampled_data'
chr_folder = 'all'
data_dir = os.path.join(input_dir, chr_folder)
response_file = os.path.join(data_dir, 'BIN50_enhancer_atlas.npy')
model_save_dir = os.path.join(input_dir, 'gradient_boosting_metropolis_hastings', chr_folder)
os.makedirs(model_save_dir, exist_ok=True)

#get X and Y
response = np.load(response_file)
for file in os.listdir(data_dir):
    if file.startswith('BIN50_') and file.endswith('.npy') and (file != 'BIN50_enhancer_atlas.npy'  and not file.startswith("BIN50_enhancer_atlas")):
        feature_name = file.split('.')[0]
        features[feature_name] = np.load(os.path.join(data_dir, file))

for key in ['promoter_forward', 'promoter_reverse']:
    if key in features:
        max_value = np.max(features[key][np.isfinite(features[key])])  # Find the max finite value
        features[key][np.isinf(features[key])] = 10 * max_value       # Replace inf with 10 times the max value


X = np.hstack([features[key].reshape(len(features[key]), -1) for key in features.keys()])
y = (response != 0).astype(int)

X = np.nan_to_num(X, nan=np.nanmean(X, axis=0), posinf=10*np.nanmax(X[np.isfinite(X)]), neginf=-10*np.nanmax(X[np.isfinite(X)]))

# Leave-One-Out Cross-Validation
chromosome_length = len(y)
num_splits = 5
split_size = chromosome_length // num_splits
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

niter = 10000



#proposal distribution probability (for one parameter)
def proposal_prob(toParam,fromParam,sigma_p):
    p = stats.norm.pdf(np.log(toParam),loc= np.log(fromParam),scale =sigma_p)
    return p

sigma_p = 2
sigma_l = 0.001

accepted_accuracies=[]
accepted_precisions=[]
accepted_recalls=[]
accepted_f1_scores=[]
accepted_roc_auc_scores=[]

model_index = 0

accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_auc_scores = []
for i, (train_index, test_index) in enumerate(kf.split(X)):

    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create and train the model
    model = GradientBoostingClassifier(n_estimators=10, random_state=42,learning_rate=0.001,min_samples_leaf=4,max_depth=10)
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(model_save_dir, f"model_{model_index}_{i}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Evaluate the model
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    roc_auc_scores.append(roc_auc_score(y_test, y_pred_prob))

accepted_accuracies.append(accuracies)
accepted_f1_scores.append(f1_scores)
accepted_precisions.append(precisions)
accepted_recalls.append(recalls)
accepted_roc_auc_scores.append(roc_auc_scores)
model_index+=1
roc_auc_average = sum(roc_auc_scores) / float(len(roc_auc_scores))
roc_auc_best = roc_auc_average
print(model_index,roc_auc_average)

###  Start iterating ###
for i in range(niters):
    # 1. propose parameters 
    learning_rate_p = np.exp(np.random.normal(np.log(learning_rate),sigmal_l))                                 # <-----------
    min_leaves_p = round(np.exp(np.random.normal(np.log(min_leaves),sigma_p)))
    max_depth_p = round(np.exp(np.random.normal(np.log(max_depth),sigma_p)))
    # 1b. Compute Proposal distributions: 
    # Q(theta|theta')
    Q_Current_from_proposed = proposal_prob(learning_rate,learning_rate_p,sigma_l)*proposal_prob(min_leaves,mean_leaves_p,sigma_p)*proposal_prob(max_depth,max_depth_p,sigma_p)
    #Q(theta'|theta)
    Q_Proposed_from_current = proposal_prob(learning_rate_p,learning_rate,sigma_l)*proposal_prob(min_leaves_p,mean_leaves,sigma_p)*proposal_prob(max_depth_p,max_depth,sigma_p)
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_auc_scores = []
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training fold {i + 1}/{num_splits}")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create and train the model
        model = GradientBoostingClassifier(n_estimators=10, random_state=42,learning_rate=learning_rate_p,min_samples_leaf=min_leaves_p,max_depth=max_depth_p)
        model.fit(X_train, y_train)

        # Save the model
        model_path = os.path.join(model_save_dir, f"model_{model_index}_{i}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Evaluate the model
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_pred_prob))

    roc_auc_average_p = sum(roc_auc_scores) / float(len(roc_auc_scores))
    
    posterior_ratio = (roc_auc_average_p/roc_auc_average)**2
    

    #3. Compute acceptance criterion
    rho = min(1,Q_Current_from_proposed/Q_Proposed_from_current*posterior_ratio)
    # 4.
    u = np.random.rand()
    
    # 5. Accept new parameters
    if u < rho:
        model_index += 1
        learning_rate = learning_rate_p
        min_leaves = mean_leaves_p
        max_depth = max_depth_p

        accepted_accuracies.append(accuracies)
        accepted_f1_scores.append(f1_scores)
        accepted_precisions.append(precisions)
        accepted_recalls.append(recalls)
        accepted_roc_auc_scores.append(roc_auc_scores)
        if roc_auc_best <roc_auc_average_p:
            roc_auc_best = roc_auc_average_p
        roc_auc_average = roc_auc_average_p

        print(model_index,roc_auc_average)