## WE USE THE HIGHER LEVEL TENSORFLOW LIBRARY CALLED TF.CONTRIB WHICH HAS AN LSTM CELL
## IMPLEMENTED. ALSO, A SOFTWARE TEMPLATE COMING WITH MIT LICENCE FOR 1 LAYER MNIST DATASET
## IMPLEMENTATION WAS USED AS AN INITIAL TEMPLATE


# https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition
# The MIT License (MIT)
#
# Copyright (c) 2016 Guillaume Chevalier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from scipy import stats
import h5py
import os
import sys
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# def feature_normalize(dataset):
#     mu = np.mean(dataset,axis = 0)
#     sigma = np.std(dataset,axis = 0)
#     return (dataset - mu)/sigma
    

def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += (size / 2)

def segment_opp(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,77))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
            # print "x_start_end",x_train[start:end]
            # segs =  x_train[start:end]
            # segments = np.concatenate((segments,segs))
            # segments = np.vstack((segments,x_train[start:end]))
            # segments = np.vstack([segments,segs])
            # segments = np.vstack([segments,x_train[start:end]])
            # labels = np.append(labels,stats.mode(y_train[start:end]))
    return segments, labels

def segment_dap(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,9))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels

def segment_pa2(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,52))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels
def segment_sph(x_train,y_train,window_size):
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size,52))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0
    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            segments[i_segment] = x_train[start:end]
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test, dataset, input_width):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        print "len(x_train[0])",len(X_train[0])
        
        # DEFINING THE MODEL
        if dataset=="opp":
            print "opp"
            self.input_height = 1
            self.input_width = input_width #or 90 for actitracker
            self.num_labels = 18  #or 6 for actitracker
            self.num_channels = 77 #or 3 for actitracker 
        elif dataset=="dap":
            print "dap"
            self.input_height = 1
            self.input_width = input_width #or 90 for actitracker
            self.num_labels = 2  #or 6 for actitracker
            self.num_channels = 9 #or 3 for actitracker
        elif dataset == "pa2":
            print "pa2"
            self.input_height = 1
            self.input_width = input_width #or 90 for actitracker
            self.num_labels = 11  #or 6 for actitracker
            self.num_channels = 52 #or 3 for actitracker
        elif dataset =="sph":
            print "sph"
            self.input_height = 1
            self.input_width = input_width #or 90 for actitracker
            self.num_labels = 20  #or 6 for actitracker
            self.num_channels = 52 #or 3 for actitracker
        else:
            print "wrong dataset"
        

        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 10
        self.batch_size = 64

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: 3 * 3D sensors features over time
        print "n_inputs len(X_train[0][0])",len(X_train[0][0])
        self.n_hidden = 64  # nb of neurons inside the neural network
        self.n_classes = self.num_labels  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, config.n_inputs])

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)

    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
    # lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.5, state_is_tuple=True)
    # lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2,lstm_cell_3], state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']




print "starting..."
start_time = time.time()

# DATA PREPROCESSING

# we start by reading the hdf5 files to a x_train variable, and return the segments to a train_x variable
# this applies for the test and validate sets as well.

if len(sys.argv)<2:
    print "Correct use:python script.py <valid_dataset>"
    sys.exit()


dataset = sys.argv[1]
if dataset == "opp":
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'OpportunityUCIDataset', 'opportunity.h5')
elif dataset =="dap":
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'dataset_fog_release','dataset_fog_release', 'daphnet.h5')
elif dataset =="pa2":
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'PAMAP2_Dataset', 'pamap2.h5')
elif dataset =="sph":
    path = os.path.join(os.path.expanduser('~'), 'Downloads', 'SphereDataset', 'sphere.h5')
else:
    print "Dataset not supported yet"
    sys.exit()

f = h5py.File(path, 'r')


x_train = f.get('train').get('inputs')[()]
y_train = f.get('train').get('targets')[()]

x_test = f.get('test').get('inputs')[()]
y_test = f.get('test').get('targets')[()]

print "x_train shape = ", x_train.shape
print "y_train shape =",y_train.shape
print "x_test shape =" ,x_test.shape
print "y_test shape =",y_test.shape

if dataset == "dap":
    # downsample to 30 Hz 
    x_train = x_train[::2,:]
    y_train = y_train[::2]
    x_test = x_test[::2,:]
    y_test = y_test[::2]
    print "x_train shape(downsampled) = ", x_train.shape
    print "y_train shape(downsampled) =",y_train.shape
    print "x_test shape(downsampled) =" ,x_test.shape
    print "y_test shape(downsampled) =",y_test.shape

if dataset == "pa2":
    # downsample to 30 Hz 
    x_train = x_train[::3,:]
    y_train = y_train[::3]
    x_test = x_test[::3,:]
    y_test = y_test[::3]
    print "x_train shape(downsampled) = ", x_train.shape
    print "y_train shape(downsampled) =",y_train.shape
    print "x_test shape(downsampled) =" ,x_test.shape
    print "y_test shape(downsampled) =",y_test.shape

print np.unique(y_train)
print np.unique(y_test)
unq = np.unique(y_test)

input_width = 23
if dataset == "opp":
    input_width = 23
    print "segmenting signal..."
    train_x, train_y = segment_opp(x_train,y_train,input_width)
    test_x, test_y = segment_opp(x_test,y_test,input_width)
    print "signal segmented."
elif dataset =="dap":
    print "dap seg"
    input_width = 25
    print "segmenting signal..."
    train_x, train_y = segment_dap(x_train,y_train,input_width)
    test_x, test_y = segment_dap(x_test,y_test,input_width)
    print "signal segmented."
elif dataset =="pa2":
    print "pa2 seg"
    input_width = 25
    print "segmenting signal..."
    train_x, train_y = segment_pa2(x_train,y_train,input_width)
    test_x, test_y = segment_pa2(x_test,y_test,input_width)
    print "signal segmented."
elif dataset =="sph":
    print "sph seg"
    input_width = 25
    print "segmenting signal..."
    train_x, train_y = segment_sph(x_train,y_train,input_width)
    test_x, test_y = segment_sph(x_test,y_test,input_width)
    print "signal segmented."
else:
    print "no correct dataset"
    exit(0)

print "train_x shape =",train_x.shape
print "train_y shape =",train_y.shape
print "test_x shape =",test_x.shape
print "test_y shape =",test_y.shape

# One-hot label conversion

train = pd.get_dummies(train_y)
test = pd.get_dummies(test_y)

train, test = train.align(test, join='inner', axis=1) # maybe 'outer' is better

train_y = np.asarray(train)
test_y = np.asarray(test)


print "unique test_y",np.unique(test_y)
print "unique train_y",np.unique(train_y)
print "test_y[1]=",test_y[1]
# test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)
print "train_y shape(1-hot) =",train_y.shape
print "test_y shape(1-hot) =",test_y.shape


config = Config(train_x, test_x, dataset, input_width)

X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
Y = tf.placeholder(tf.float32, [None, config.n_classes])

pred_Y = LSTM_Network(X, config)

# Loss,optimizer,evaluation
l2 = config.lambda_loss_amount * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# Softmax loss and L2
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) #+ l2
# optimizer = tf.train.AdagradOptimizer(
#     learning_rate=config.learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(
    learning_rate=config.learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


training_epochs = 200
loss_over_time = np.zeros(training_epochs)
total_batches = train_x.shape[0] // config.batch_size
b = 0
# Launch the graph
with tf.Session() as sess:
    # sess.run(init)
    tf.initialize_all_variables().run()
    # Keep training until reach max iterations
    # cost_history = np.empty(shape=[0],dtype=float)
    for epoch in range(training_epochs):
        cost_history = np.empty(shape=[0],dtype=float)
        for b in range(total_batches):
            offset = (b * config.batch_size) % (train_y.shape[0] - config.batch_size)
            batch_x = train_x[offset:(offset + config.batch_size), :, :]
            batch_y = train_y[offset:(offset + config.batch_size), :]
            
            # print "batch_x shape =",batch_x.shape
            # print "batch_y shape =",batch_y.shape

            _, c = sess.run([optimizer, cost],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)
        loss_over_time[epoch] = np.mean(cost_history)
        print "Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y})

    # MORE METRICS
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
    # pred_Y is the result of the FF-RNN
    y_p = tf.argmax(pred_Y, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={X:test_x, Y:test_y})
    print "validation accuracy:", val_accuracy
    y_true = np.argmax(test_y,1)
    # print "Precision,micro", metrics.precision_score(y_true, y_pred,average="micro")
    # print "Precision,macro", metrics.precision_score(y_true, y_pred,average="macro")
    # print "Precision,weighted", metrics.precision_score(y_true, y_pred,average="weighted")
    # #print "Precision,samples", metrics.precision_score(y_true, y_pred,average="samples")
    # print "Recall_micro", metrics.recall_score(y_true, y_pred, average="micro")
    # print "Recall_macro", metrics.recall_score(y_true, y_pred, average="macro")
    # print "Recall_weighted", metrics.recall_score(y_true, y_pred, average="weighted")
    # #print "Recall_samples", metrics.recall_score(y_true, y_pred, average="samples")
    # print "f1_score_micro", metrics.f1_score(y_true, y_pred, average="micro")
    # print "f1_score_macro", metrics.f1_score(y_true, y_pred, average="macro")
    # print "f1_score_weighted", metrics.f1_score(y_true, y_pred, average="weighted")
    #print "f1_score_samples", metrics.f1_score(y_true, y_pred, average="samples")
    if dataset=="opp" or dataset == "pa2" :
        #print "f1_score_mean", metrics.f1_score(y_true, y_pred, average="micro")
        print "f1_score_w", metrics.f1_score(y_true, y_pred, average="weighted")
        
        print "f1_score_m", metrics.f1_score(y_true, y_pred, average="macro")
        # print "f1_score_per_class", metrics.f1_score(y_true, y_pred, average=None)
    elif dataset=="dap":
        print "f1_score_m", metrics.f1_score(y_true, y_pred, average="macro")
    elif dataset=="sph":
        print "f1_score_mean", metrics.f1_score(y_true, y_pred, average="micro")
        print "f1_score_w", metrics.f1_score(y_true, y_pred, average="weighted")
        
        print "f1_score_m", metrics.f1_score(y_true, y_pred, average="macro")
    else:
        print "wrong dataset"
    # if dataset=="dap":
    #     print "f1_score",metrics.f1_score(y_true, y_pred)
    print "confusion_matrix"
    print metrics.confusion_matrix(y_true, y_pred)
    # plt.figure(1)
    # plt.plot(loss_over_time)
    # plt.title("Loss value over epochs (FFLSTM DG)")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

#######################################################################################
#### micro- macro- weighted explanation ###############################################
#                                                                                     #
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html      #
#                                                                                     #
# micro :Calculate metrics globally by counting the total true positives,             #
# false negatives and false positives.                                                #
#                                                                                     #
# macro :Calculate metrics for each label, and find their unweighted mean.            #
# This does not take label imbalance into account.                                    #
#                                                                                     #
# weighted :Calculate metrics for each label, and find their average, weighted        #
# by support (the number of true instances for each label). This alters macro         #
# to account for label imbalance; it can result in an F-score that is not between     #
# precision and recall.                                                               #
#                                                                                     #
#######################################################################################


print("--- %s seconds ---" % (time.time() - start_time))
print "done."
