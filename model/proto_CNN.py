# Proto HMM
# UNDER CONSTRUCTION

# Imports, temporal
import os
import numpy as np
import tensorflow as tf
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import Sequential

from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout

# Globals?
# Ranges to analyze
chr_no = "chr1"
demo_start = 0
#demo_end = 20000000 # 20M
demo_end = 500000 # 500k
demo_len = demo_end-demo_start


# READ BASE PAIR SEQUENCE FROM DATA FOLDER, CHOOSING CHROMOSOME NUMBER, RANGE (ALL BY DEFAULT) AND REF GENOME (HG19 BY DEFAULT)
def read_chromosome(chromosome_number, ref_genome="hg19", range_start=None, range_end=None):
    file_path = f"../data/{ref_genome}/{chromosome_number}.fa"
    with open(file_path, "r") as g:
        g.readline()  # Skip the header line
        lines = g.readlines()
        print("Lines are read")

    #length = len(lines)
    if range_start==None:
        range_start=0
    if range_end==None:
        range_start=len(lines)-1
    
    # Combine all lines into a single string and strip newlines
    full_chromosome_data = ''.join(line.strip() for line in lines)

    # Determine the length of the chromosome data
    chromosome_length = len(full_chromosome_data)
    
    # Set default values for range_start and range_end
    if range_start is None:
        range_start = 0
    if range_end is None:
        range_end = chromosome_length

    # Validate and adjust range_end if it exceeds the chromosome length
    if range_end > chromosome_length:
        range_end = chromosome_length

    step = (range_end - range_start) // 20 # Print progress every 5%

    # Extract the desired range of chromosome data
    chromosome_data = ""
    for i in range(range_start, range_end):
        chromosome_data += full_chromosome_data[i]
        
        # Print out progress
        if (i - range_start) % step == 0:
            print(f"{((i - range_start) / (range_end - range_start)) * 100:.2f}% read")

    print("Chromosome data extracted")

    return chromosome_data


##################################################################
# LOAD GENOME DATA from data folder into variable Genome, 
# which is a list of numbers from the dictionary dDNA
start_gen_time = time.time()
Genome_letters=read_chromosome(chr_no, range_start=demo_start, range_end=demo_end)
print("\n--- Reading in specified range took %s seconds ---\n" % (time.time() - start_gen_time))

Genome_l=list(Genome_letters)
# Transform letters to numbers
dDNA = {"a": 0 , "c": 1 , "g": 2, "t": 3, "n": -1,
         "A": 0 , "C": 1 , "T": 2, "G": 3, "N": -1}
Genome = np.array([dDNA[bp] for bp in Genome_l])
##################################################################

##################################################################
# READ ANNOTATION DATA
def load_database(database_name, file_ext=".txt"):
    database_path = "../data/" + database_name
    database = {}

    # Iterate over all files in the directory
    for filename in os.listdir(database_path):
        if filename.endswith(file_ext):
            cell_line = filename.split(".")[0]  # Extract the cell line name from the filename
            file_path = os.path.join(database_path, filename)

            with open(file_path, "r") as f:
                data_entries = []
                for line in f.readlines():
                    line = line.strip().split("\t")
                    data_entries.append([line[0], int(line[1]), int(line[2]), float(line[3])])

                # Store the data in the dictionary
                database[cell_line] = data_entries

    return database
# OUTPUT: database["cell_line"][line_no][item_no], where item_no are 0:chromosome_no, 1:start, 2: end, 3:enrichment_score

# We load enhancer atlas, for example, and specifically GM12878
enhancer_atlas_db = load_database("enhancer_atlas")

# Keep only enhancers inside chromosome and range
ranged_cell_line = []
cur_cell_line=enhancer_atlas_db["GM12878"]
for i in range(len(cur_cell_line)):
    # Check chromosome number, and range of enhancer
    if (cur_cell_line[i][0] == chr_no) and (demo_start <= cur_cell_line[i][1] <= demo_end and demo_start <= cur_cell_line[i][2] <= demo_end):
        ranged_cell_line.append(cur_cell_line[i])


# Multidimensional annotation
Annotation=[[]]

# FOR NOW WE WILL DO 0-1
# Set all annotations to 0
for i in range(len(Genome)):
    Annotation[0].append(0)
# Set enhancer regions to 1
for i in range(len(ranged_cell_line)):
    cur_enha=ranged_cell_line[i]
    for j in range(cur_enha[1], cur_enha[2]):
        Annotation[0][j] = 1

# Adding an annotation layer would be:
#Annotation.append([])
# Accessing new annotation layer:
#Annotation[1]

cur_annotation = Annotation[0]
##################################################################

##################################################################
## Split into halves for train and test
genome_train = Genome[:(demo_len//2)]
genome_test = Genome[(demo_len//2):]
annotation_train = cur_annotation[:(demo_len//2)]
annotation_test = cur_annotation[(demo_len//2):]
##################################################################

##################################################################
## Just some interesting data:
print("Random, very deletable data:\n length genome:", len(Genome),
       "\n length 1st annotation:", len(Annotation[0]), # Just to check they're the same length
        "\n nº enhancers in range:", len(ranged_cell_line), 
        "\n nº total enhancers this cell line:", len(cur_cell_line))
##################################################################

## DEFINING 2 MODELS
class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32,3,padding='same',activation='relu')
        self.pool1 = MaxPool2D((2,2))
        self.conv2 = Conv2D(64,3,padding='same',activation='relu')
        self.pool2 = MaxPool2D((2,2))
        self.flatten = Flatten()
        self.d1 = Dense(512,activation='relu')
        self.dropout1 = Dropout(0.4)
        self.d2 = Dense(128,activation= 'relu')
        self.dropout2 = Dropout(0.4)
        self.d3 = Dense(2,activation = 'softmax')

    def call(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.dropout2(x)
        x = self.d3(x)
        return x 


class NN(Model):
    def __init__(self):
        super(NN, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(32,activation='relu')
        self.dropout1 = Dropout(0.4)
        self.d2 = Dense(128,activation= 'relu')
        self.dropout2 = Dropout(0.4)
        self.d3 = Dense(2,activation = 'softmax')

    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.dropout2(x)
        x = self.d3(x)
        return x 

myCNN = CNN()
myNN = NN()

## DEFINING PERFORMANCE VARIABLES
loss_object = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
recall1 = tf.keras.metrics.Recall(name = "recall")
prec1 = tf.keras.metrics.Precision(name = "precicion")

f_pos1 = tf.keras.metrics.FalsePositives(name = "false_positives")
t_pos1 = tf.keras.metrics.TruePositives(name = "true_positives")
t_neg1 = tf.keras.metrics.TrueNegatives(name = "true_negatives")
f_neg1 = tf.keras.metrics.FalseNegatives(name = "false_negatives")

# 2nd model
loss_object2 = tf.keras.losses.CategoricalCrossentropy()
optimizer2 = tf.keras.optimizers.Adam()
train_loss2 = tf.keras.metrics.Mean(name="train_loss")
train_accuracy2 = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
recall2 = tf.keras.metrics.Recall(name = "recall")
prec2 = tf.keras.metrics.Precision(name = "precicion")

f_pos2 = tf.keras.metrics.FalsePositives(name = "false_positives")
t_pos2 = tf.keras.metrics.TruePositives(name = "true_positives")
t_neg2 = tf.keras.metrics.TrueNegatives(name = "true_negatives")
f_neg2 = tf.keras.metrics.FalseNegatives(name = "false_negatives")

## DEFINING TRAINING FUNCTIONS
def train_step(images,labels,model):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels,predictions)
    gradients= tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels,predictions)
    recall1(labels,predictions)
    prec1(labels, predictions)
    f_pos1(labels, predictions)
    t_pos1(labels, predictions)
    t_neg1(labels, predictions)
    f_neg1(labels, predictions)
    
# 2nd model (we will see)
def train_step2(images,labels,model):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object2(labels,predictions)
    gradients= tape.gradient(loss,model.trainable_variables)
    optimizer2.apply_gradients(zip(gradients,model.trainable_variables))

    train_loss2(loss)
    train_accuracy2(labels,predictions)
    recall2(labels,predictions)
    prec2(labels, predictions)
    f_pos2(labels, predictions)
    t_pos2(labels, predictions)
    t_neg2(labels, predictions)
    f_neg2(labels, predictions)


EPOCHS = 5
# Probably optional, if we're not gonna use more than 1 model
n_models = 2

### METRICS ###
l_loss = []
l_accuracy = []
l_recall = []
l_prec = []
l_f_pos = []
l_t_pos = []
l_t_neg = []
l_f_neg = []

# As said, probably optional
for models in range(n_models):
    l_loss.append([])
for models in range(n_models):
    l_accuracy.append([])
for models in range(n_models):
    l_recall.append([])
for models in range(n_models):
    l_prec.append([])
for models in range(n_models):
    l_f_pos.append([])
for models in range(n_models):
    l_t_pos.append([])
for models in range(n_models):
    l_t_neg.append([])
for models in range(n_models):
    l_f_neg.append([])
### END OF METRICS ###

window_size = 800
step_size = 1

# Convert annotations to one-hot encoding for classification (2 classes: 0 and 1)
#annotation_train_one_hot = tf.keras.utils.to_categorical(annotation_train, num_classes=2)

print("\nTraining with CNN")
for epoch in range(EPOCHS):
    # Iterate over each sequence in the training dataset
    dataset_length = len(genome_train)
    if dataset_length >= window_size:
        # Iterate over the dataset with the sliding window
        for start in range(0, dataset_length - window_size + 1, step_size):
            end = start + window_size
            window = genome_train[start:end]
            window_labels = annotation_train[start:end]

            # Prepare window and labels as a batch
            window_batch = np.array(window).reshape(1, window_size, 1)  # Reshape to match CNN input
            label_batch = np.array(window_labels).reshape(1, window_size, 2)  # Reshape to match label format

            # Feed the window and labels to the train_step function
            train_step(window_batch, label_batch, myCNN)
    #myCNN.save_weights('/weights',save_format='tf')

    l_loss[0].append(train_loss.result().numpy())
    l_accuracy[0].append(train_accuracy.result().numpy()*100)
    l_recall[0].append(recall1.result().numpy()*100)
    l_prec[0].append(prec1.result().numpy()*100)
    l_f_pos[0].append(f_pos1.result().numpy())
    l_t_pos[0].append(t_pos1.result().numpy())
    l_t_neg[0].append(t_neg1.result().numpy())
    l_f_neg[0].append(f_neg1.result().numpy())

    print("EPOCH: ",str(epoch+1),' Loss:',str(l_loss[0][epoch]),' Precision: ', str(l_prec[0][epoch]), ' False positives: ', str(l_f_pos[0][epoch]), ' True positives: ', str(l_t_pos[0][epoch]), ' True negatives: ', str(l_t_neg[0][epoch]), ' False negatives: ', str(l_f_neg[0][epoch]))
    train_loss.reset_states()
    train_accuracy.reset_states()
    recall1.reset_state()
    prec1.reset_state()
    f_pos1.reset_state()
    t_pos1.reset_state()
    t_neg1.reset_state()
    f_neg1.reset_state()

# Working on no1 for now
# print("\nTraining with Non-Convolutional Neural Network", "\"NN\"")
# for epoch in range(EPOCHS):
#     for images, labels in train_ds:
#         train_step2(images,labels,myNN)
#     #myNN.save_weights('/weights',save_format='tf')

#     l_loss[1].append(train_loss2.result().numpy())
#     l_accuracy[1].append(train_accuracy2.result().numpy()*100)
#     l_recall[1].append(recall2.result().numpy()*100)
#     l_prec[1].append(prec2.result().numpy()*100)
#     l_f_pos[1].append(f_pos2.result().numpy())
#     l_t_pos[1].append(t_pos2.result().numpy())
#     l_t_neg[1].append(t_neg2.result().numpy())
#     l_f_neg[1].append(f_neg2.result().numpy())

#     print("EPOCH: ",str(epoch+1),' Loss:',str(l_loss[1][epoch]),' Precision: ', str(l_prec[1][epoch]), ' False positives: ', str(l_f_pos[1][epoch]), ' True positives: ', str(l_t_pos[1][epoch]), ' True negatives: ', str(l_t_neg[1][epoch]), ' False negatives: ', str(l_f_neg[1][epoch]))
#     train_loss2.reset_states()
#     train_accuracy2.reset_states()
#     recall2.reset_state()
#     prec2.reset_state()
#     f_pos2.reset_state()
#     t_pos2.reset_state()
#     t_neg2.reset_state()
#     f_neg2.reset_state()



##################################################################
## Creating, and printing confusion matrix-related data
#np.savetxt('prediction.txt', pred, delimiter=',', fmt='%0.3f')

def create_confusion_matrix(actual, predict):
    TP = 0 # cm[0,0]
    FP = 0 # cm[0,1]
    FN = 0 # cm[1,0]
    TN = 0 # cm[1,1]

    for i in range(len(actual)):
        if actual[i]==predict[i]:
            if actual[i]==1:
                TP+=1
            else:
                TN+=1
        elif actual[i]==1:
            FN+=1
        else:
            FP+=1

    return np.array([[TP, FP], [FN, TN]])

#cm = create_confusion_matrix(annotation_test, pred)

#print("TP:", cm[0,0], "FP:", cm[0,1], "FN:", cm[1,0], "TN:", cm[1,1])
#print("Accuracy:", ((cm[0,0] + cm[1,1])/np.sum(cm))*100, "%")
##################################################################