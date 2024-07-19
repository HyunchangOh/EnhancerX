# MLP

#########################################################################################################
## Imports

# The Python Standard Library
import os
import numpy as np
import time

# Tensorflow and scikitlearn for trand and test splitting, confusion matrix, and ppv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score

#########################################################################################################


#########################################################################################################
## Globals

save_res = "../../Subsampled_Final/NN_results/"
chromosomes = [
        "chr1",
        "chr2",
        "chr3",
        "chr4",
        "chr5",
        "chr6",
        "chr7",
        "chr8",
        "chr9",
        "chr10",
        "chr11",
        "chr12",
        "chr13",
        "chr14",
        "chr15",
        "chr16",
        "chr17",
        "chr18",
        "chr19",
        "chr20",
        "chr21",
        "chr22",
        "chrX",
        "chrY"
    ]

# Complete list of features
features = [
    "BIN50_cod",
    "BIN50_cod_1D_Dist",

    "BIN50_promoter_any", #2
    "BIN50_promoter_1D_Dist_any",
    "BIN50_promoter_3D_Dist_any",
    "BIN50_promoter_forward",
    "BIN50_promoter_1D_Dist_forward",
    "BIN50_promoter_reverse",
    "BIN50_promoter_1D_Dist_reverse",

    "BIN50_h3k4me1", #9
    "BIN50_h3k4me1_1D_Dist",
    "BIN50_h3k4me1_3D_Dist",

    "BIN50_h3k4me2", #12
    "BIN50_h3k4me2_1D_Dist",
    "BIN50_h3k4me2_3D_Dist",

    "BIN50_h3k9me3", #15
    "BIN50_h3k9me3_1D_Dist",
    "BIN50_h3k9me3_3D_Dist",

    "BIN50_h3k27ac", #18
    "BIN50_h3k27ac_1D_Dist",
    "BIN50_h3k27ac_3D_Dist",

    "BIN50_h3k27me3", #21
    "BIN50_h3k27me3_1D_Dist",
    "BIN50_h3k27me3_3D_Dist",

    "BIN50_h3k36me3", #24
    "BIN50_h3k36me3_1D_Dist",
    "BIN50_h3k36me3_3D_Dist",

    "BIN50_CTCF", #27
    "BIN50_CTCF_1D_Dist",
    "BIN50_CTCF_3D_Dist",
    
    "BIN50_DHS", #30
    "BIN50_DHS_1D_Dist",
    "BIN50_DHS_3D_Dist",

    "BIN50_EP300Conservative", #33
    "BIN50_EP300Conservative_1D_Dist",
    "BIN50_EP300Conservative_3D_Dist",

    "BIN50_Interaction_1D_Dist" #36
    ]

# Selected features because of dependency, redundancy, etc
feature_dist = [
    # Promoter
    "BIN50_promoter_3D_Dist_any",

    # Coding/non-coding
    "BIN50_cod_1D_Dist",

    # Epigenetics
    "BIN50_h3k4me1_3D_Dist",
    "BIN50_h3k4me2_3D_Dist",    
    "BIN50_h3k9me3_3D_Dist",
    "BIN50_h3k27ac_3D_Dist",
    "BIN50_h3k27me3_3D_Dist",
    "BIN50_h3k36me3_3D_Dist",
    "BIN50_CTCF_3D_Dist",
    "BIN50_DHS_3D_Dist",
    "BIN50_EP300Conservative_3D_Dist"
    ]
feature_mem = [
    # Promoter
    "BIN50_promoter_any",

    # Coding/non-coding
    "BIN50_cod",

    # EpigeneticS
    "BIN50_h3k4me1", #9
    "BIN50_h3k4me2", #12
    "BIN50_h3k9me3", #15
    "BIN50_h3k27ac", #18
    "BIN50_h3k27me3", #21
    "BIN50_h3k36me3", #24
    "BIN50_CTCF", #27
    "BIN50_DHS", #30
    "BIN50_EP300Conservative" #33
    ]
feature_mix = [
    # Promoter
    "BIN50_promoter_3D_Dist_any",

    # Coding/non-coding
    "BIN50_cod",

    # EpigeneticS
    "BIN50_h3k4me1", #9
    "BIN50_h3k4me2", #12
    "BIN50_h3k9me3", #15
    "BIN50_h3k27ac", #18
    "BIN50_h3k27me3", #21
    "BIN50_h3k36me3", #24
    "BIN50_CTCF", #27
    "BIN50_DHS", #30
    "BIN50_EP300Conservative" #33
    ]
    
# Model default parameters
validation = "BIN50_enhancer_atlas"
num_epochs = 6
batch_size = 32

# List of structures to try, changeable
structures = [
    [32, 32],
    [64, 64],
    #[128, 128],
    [32, 32, 32],
    [64, 64, 64],
    #[128, 128, 128],
    #[64, 32, 16],
    [32, 32, 32, 32]
]

#########################################################################################################

#########################################################################################################
## Standardized loading function. Take feature list as input.

def mucho_load(chromosome_number, features_list,  validation = "BIN50_enhancer_atlas"):

    file_path = "../../Subsampled_Final/" + chromosome_number + "/"
    
    mucho_time = time.time()

    data_length = np.load(file_path + validation + ".npy", mmap_mode="r").size
    no_col = len(features_list) + 1

    # Check for size comparing all columns to validation data length
    check = [np.load(file_path + feature + ".npy", mmap_mode="r").size for feature in features_list]
    if any(size != data_length for size in check):
        print(next(feature for feature, size in zip(features_list, check) if size != check[0]) + " is not of data length.")

    # Initialize the la_grande_table with the correct shape
    la_grande_table = np.empty((no_col, data_length))

    # Load validation as the first column and then fill the rest
    la_grande_table[0] = np.load(file_path + validation + ".npy")
    for idx, feature in enumerate(features_list, start = 1):
        la_grande_table[idx] = np.load(file_path + feature + ".npy")

    print("\n--- Loaded la grande table for " + chromosome_number + ": " + str(la_grande_table.shape[0]) + " columns, " + str(la_grande_table.shape[1]) + " items in %s seconds ---\n" % (time.time() - mucho_time))

    return la_grande_table

# Use, before any model, would be:
# np_array = mucho_load("chr1", feature_list)
# Access table: la_grande_table[col][position(s)]

#########################################################################################################


#########################################################################################################
## Function to construct the Neural Network model.

def build_model(input_shape, structure):

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(structure[0], activation="relu", input_shape = (input_shape,)))
    for i in range(1, len(structure)):
        model.add(tf.keras.layers.Dense(structure[i], activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    return model

#########################################################################################################


#########################################################################################################
## Model definition and running.

def run_model(chromosome_number, features_list, num_epochs = 5, batch_size = 32, validation = "BIN50_enhancer_atlas", structures = [[32, 32, 32]], lrn_rate = 0.001, f_list = "feature_mix"):

    # Load la grande table, with column 0 being the validation data
    la_mesa = mucho_load(chromosome_number, features_list, validation = validation)

    # Transpose to make each row a bin and each column a feature
    data = la_mesa.T

    # Assuming that the "enhancer_atlas" column (first column in this case) is the target variable
    X = np.delete(data, 0, axis=1)  # All features except the enhancer column
    y = data[:, 0]  # The enhancer column

    # Apply threshold to y, which for now it"s 0, so if a bin has 1 or more enhancer associated base pairs, it classifies as enhancer
    y = (y > 0).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # List to store the results
    acc_list = []
    ppv_list = []
    auc_list = []

    prel_res = {
        "acc": [],
        "ppv": [],
        "auc": []
    }

    chr_time = time.time()

    for structure in structures:

        # Build the model
        model = build_model(X_train.shape[1], structure)

        # Compile the model
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lrn_rate),
                    loss = "binary_crossentropy",
                    metrics=["accuracy"])

        # Train the model
        model_time = time.time()
        model.fit(x = X_train, y = y_train, epochs = num_epochs, batch_size = batch_size, validation_split = 0.2, verbose = 0) # Unless we need to print epochs

        print("\n--- Ran structure", structure, " in %s seconds ---" % round(time.time() - model_time))

        # Evaluate the model
        results = model.evaluate(X_test, y_test, verbose = 0)
        result_dict = {name: result for name, result in zip(model.metrics_names, results)}
        acc_list.append((structure, result_dict["accuracy"]))

        # Make predictions on the test set, classify more than 0.5 as enh, less as non-enh, from sigmoid act function
        predictions = model.predict(X_test)
        binary_predictions = (predictions > 0.5).astype(int)

        # Save test vs predictions to a npy for later assesment if necessary
        y_test = np.array(y_test)
        save_res = "../../Subsampled_Final/NN_results/"
        file_name = "testvpred" + "_" + str(structure) + "_" + str(batch_size) + "_" + str(lrn_rate) + "_" + str(num_epochs)
        np.save((save_res + chromosome_number + "/" + f_list + "/" + file_name + ".npy"), np.column_stack((y_test, binary_predictions)))
        
        ppv = precision_score(y_test, binary_predictions)
        ppv_list.append(ppv)

        # Print confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, binary_predictions).ravel()
        print("Confusion Matrix:")
        print("TP: %05d FP: %05d" % (tp, fp))
        print("FN: %05d TN: %05d" % (fn, tn))

    # Sort the results based on accuracy and ppv
    sorted_acc = sorted(acc_list, key=lambda x: x[1], reverse=True)
    sorted_ppv = sorted(list(zip(structures, ppv_list)), key=lambda x: x[1], reverse=True)

    # Print the sorted results
    print("\nSorted results based on accuracy:")
    for structure, accuracy in sorted_acc:
        print(structure, "Accuracy:", accuracy)
    
    print("\nSorted results based on ppv:")
    for structure, ppv_v in sorted_ppv:
        print(structure, "PPV:", ppv_v)
    
    # Add results to dictionary of lists:
    prel_res["acc"].append(sorted_acc[0])
    prel_res["ppv"].append(sorted_ppv[0])
    
    # want to maybe return a list with results, maybe only top structure results
    return prel_res

#########################################################################################################


#########################################################################################################
## Function to custom run, print time, etc.

def custom_run (chromosome_number, feature_list, num_epochs = 5, batch_size = 32, validation = "BIN50_enhancer_atlas", structures = [[32, 32, 32]], lrn_rate = 0.001, f_list = "feature_mix"):
    print("\n-------------------------------- Running " + chromosome_number + ": " + str(num_epochs) + " epochs, " + str(batch_size) + " batch size, " + str(lrn_rate) + " learning rate  --------------------------------")
    nn_time = time.time()
    all_res = run_model(chromosome_number, feature_list, num_epochs = num_epochs, batch_size = batch_size, validation = validation, structures = structures, lrn_rate = lrn_rate, f_list = f_list)
    print("\n------------------------ Ran NN on " + chromosome_number + ": " + str(num_epochs) + " epochs, " + str(batch_size) + " batch size, " + str(lrn_rate) + " learning rate in %s seconds ------------------------\n" % round(time.time() - nn_time))
    return all_res

#########################################################################################################


#########################################################################################################

#########################################################################################################
#######################################---- CUSTOM CUSTOM RUN ----#######################################
#########################################################################################################

def tryout(chromosome_number, feature_list, structures, num_epochs, batch_size, lrn_rate, f_list):
    
    omni_time = time.time()
    result_list1 = []
    print("\n--------------------------------------------------------------")
    print("-------------------  Trying learning rates -------------------")
    print("--------------------------------------------------------------\n")
    for i in lrn_rate:
        cur_result = custom_run(chromosome_number, feature_list = feature_list, lrn_rate = i, structures = structures, f_list = f_list)
        result_list1.append(cur_result)

    j = 0
    for i in lrn_rate:
        if j > 0 : 
            print("\n")
        print("Top results for learning rate " + str(i) + ":")
        print("Top accuracy:", result_list1[j]["acc"])
        #print("Top AUC:", result_list1[j]["auc"])
        print("Top PPV:", result_list1[j]["ppv"])
        j += 1


    result_list1 = []
    print("\n--------------------------------------------------------------")
    print("---------------------  Trying batch size ---------------------")
    print("--------------------------------------------------------------\n")

    for i in batch_size:
        cur_result = custom_run(chromosome_number, feature_list = feature_list, batch_size = i, structures = structures, f_list = f_list)
        result_list1.append(cur_result)

    j = 0
    for i in batch_size:
        if j > 0 : 
            print("\n")
        print("Top results for batch size " + str(i) + ":")
        print("Top accuracy:", result_list1[j]["acc"])
        #print("Top AUC:", result_list1[j]["auc"])
        print("Top PPV:", result_list1[j]["ppv"])
        j += 1


    result_list1 = []
    print("\n--------------------------------------------------------------")
    print("---------------------  Trying num epochs ---------------------")
    print("--------------------------------------------------------------\n")

    for i in num_epochs:
        cur_result = custom_run(chromosome_number, feature_list = feature_list, num_epochs = i, structures = structures, f_list = f_list)
        result_list1.append(cur_result)

    j = 0
    for i in num_epochs:
        if j > 0 : 
            print("\n")
        print("Top results for number of epochs " + str(i) + ":")
        print("Top accuracy:", result_list1[j]["acc"])
        print("Top PPV:", result_list1[j]["ppv"])
        j += 1

    print("\n" + "#"*90)
    print("#"*90)
    print("------------------------------------------------------------------------------------------")
    print("\nEnd of program, ran " + f_list + " for " + chromosome_number + " in %s seconds" % round(time.time() - omni_time))
    print("\n------------------------------------------------------------------------------------------")
    print("#"*90)
    print(("#"*90) + "\n")

try_learn_rate = [0.001, 0.0001, 0.00001]
try_batch_size = [32, 64, 128]
try_epochs = [5, 10, 15]

# First run on chromosome all, which is a concatenation of all chromosomes
tryout("all", feature_mix, structures, try_epochs, try_batch_size, try_learn_rate, "feature_mix")
tryout("all", feature_mem, structures, try_epochs, try_batch_size, try_learn_rate, "feature_mem")
tryout("all", feature_dist, structures, try_epochs, try_batch_size, try_learn_rate, "feature_dist")

# And then run on all chromosomes
for i in chromosomes[:-1]:
    tryout(i, feature_mix, structures, try_epochs, try_batch_size, try_learn_rate, "feature_mix")
    tryout(i, feature_mem, structures, try_epochs, try_batch_size, try_learn_rate, "feature_mem")
    tryout(i, feature_dist, structures, try_epochs, try_batch_size, try_learn_rate, "feature_dist")
