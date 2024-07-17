# Enhancer Region Prediction Models

This folder implements various machine learning models for predicting enhancer regions from genomic annotations. 
The implementation is done in Python with using sckitlearn, TensorFlow/Keras.

## Implemented Models 

This project features several well-established machine learning algorithms tailored for enhancer region prediction: 

- **Logistic Regression:** '''tolerance = 0.0001, penalty = l2, C(inverse of regularization strength) = 1.0, max_iter =100'''
- **Gradient Boosting:** '''num_estimators(trees)=10, learning_rate = 0.1, min_samples_split=2, min_samples_leaf=1, max_depth = 3'''
- **Random Forest:** '''num_estimators(trees)=10, min_samples_split=2, min_samples_leaf=1, max_depth = None'''
- **Multilayer Perceptron (MLP):** '''hidden_layer_sizes=(32-64), learning_rate =0.01-0.0001, batch_size = 32-128'''
