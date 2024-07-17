# Enhancer Region Prediction Models

This folder implements various machine-learning models for predicting enhancer regions from genomic annotations. 
The implementation is done in Python using sckitlearn, TensorFlow/Keras.

## Implemented Models 

This project features several well-established machine-learning algorithms tailored for enhancer region prediction: 
### Logistic Regression 
'''tolerance = 0.0001, penalty = l2, C(inverse of regularization strength) = 1.0, max_iter =100'''

### Gradient Boosting 
'''num_estimators(trees)=10, learning_rate = 0.1, min_samples_split=2, min_samples_leaf=1, max_depth = 3'''

### Random Forest 

#### Data Loading and Preprocessing:
 - Infinite values in promoter_forward and promoter_reverse are replaced with 10 times the maximum finite value.
 - Categorical feature seq (if exists) is one-hot encoded.
 - All features are stacked horizontally to create the feature matrix X.
 - NaNs and infinite values in X are replaced with mean values and large finite values, respectively.

#### Model Definition and Training:
 - A Random Forest model is defined and trained using 5-fold cross-validation.
 - The trained models for each fold are saved as pickle files.
 - Metrics such as accuracy, precision, recall, F1-score, and ROC AUC are calculated for each fold and saved.

#### Metrics
 - Number of Trees: 100
 - Tree Depth: The default setting of None is used, meaning nodes are expanded until all leaves are pure or contain less than the minimum samples required for splitting.
 - Splitting Criteria: Gini impurity is used as the measure for splitting nodes.

### Multilayer Perceptron (MLP) '''hidden_layer_sizes=(32-64), learning_rate =0.01-0.0001, batch_size = 32-128'''
