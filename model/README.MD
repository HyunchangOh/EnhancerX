# Enhancer Region Prediction Models

This folder implements various machine-learning models for predicting enhancer regions from genomic annotations. 
The implementation is done in Python using sckitlearn, TensorFlow/Keras.

## Implemented Models 

This project features several well-established machine-learning algorithms tailored for enhancer region prediction: 
### Logistic Regression 
'''tolerance = 0.0001, penalty = l2, C(inverse of regularization strength) = 1.0, max_iter =100'''

### Gradient Boosting 
'''num_estimators(trees)=10, learning_rate = 0.1, min_samples_split=2, min_samples_leaf=1, max_depth = 3'''

### Random Forest '''num_estimators(trees)=10, min_samples_split=2, min_samples_leaf=1, max_depth = None'''

#### Data Loading and Preprocessing:
-Loads response variable (response) indicating enhancer regions.
-Loads multiple feature data files representing genomic annotations (features).
-Handles potential issues like missing values (NaNs) and infinities in the data.
-Converts categorical variables (if present) using one-hot encoding.
-Stacks the features and response variable into matrices (X and y).

#### Model Definition and Training:
-Defines a create_rf_model() function to create a Random Forest classifier with 100 trees and a fixed random state for reproducibility.
-Implements Leave-One-Out Cross-Validation (LOOCV) with 5 splits to evaluate the model's performance on different subsets of the data.
-Within each fold:
--Splits data into training and testing sets (X_train, X_test, y_train, y_test).
--Trains the Random Forest model on the training data.
--Saves the trained model (model) using pickle.
--Evaluates the model on the testing data using various metrics (accuracy, precision, recall, F1-score, ROC AUC).
--Prints the classification report and ROC AUC for each fold.

#### Metrics and Saving:
-Stores the calculated metrics (accuracy, precision, recall, f1, roc_auc) for each fold in a dictionary (metrics).
-Saves the dictionary containing the metrics using pickle.

### Multilayer Perceptron (MLP) '''hidden_layer_sizes=(32-64), learning_rate =0.01-0.0001, batch_size = 32-128'''
