## WE USE THE HIGHER LEVEL TENSORFLOW LIBRARY CALLED TF.CONTRIB WHICH HAS AN LSTM CELL
## IMPLEMENTED. ALSO, A SOFTWARE TEMPLATE FOR 1 LAYER MNIST DATASET
## IMPLEMENTATION WAS USED AS AN INITIAL TEMPLATE Project: https://github.com/aymericdamien/TensorFlow-Examples/

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pandas as pd
from scipy import stats
import time
from sklearn import metrics
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


print "train_x shape =",train_x.shape
print "train_y shape =",train_y.shape
print "test_x shape =",test_x.shape
print "test_y shape =",test_y.shape

# 1-hot labeling
# train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
# test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)

# http://fastml.com/how-to-use-pd-dot-get-dummies-with-the-test-set/

train = pd.get_dummies(train_y)
test = pd.get_dummies(test_y)

train, test = train.align(test, join='inner', axis=1) # maybe 'outer' is better

train_y = np.asarray(train)
test_y = np.asarray(test)


print "unique test_y",np.unique(test_y)
print "unique train_y",np.unique(train_y)
print "test_y[1]=",test_y[1]
print "train_y shape(1-hot) =",train_y.shape
print "test_y shape(1-hot) =",test_y.shape


# DEFINING THE MODEL
if dataset=="opp":
    print "opp"
    input_height = 1
    input_width = input_width #or 90 for actitracker
    num_labels = 18  #or 6 for actitracker
    num_channels = 77 #or 3 for actitracker 
elif dataset=="dap":
    print "dap"
    input_height = 1
    input_width = input_width #or 90 for actitracker
    num_labels = 2  #or 6 for actitracker
    num_channels = 9 #or 3 for actitracker
elif dataset == "pa2":
    print "pa2"
    input_height = 1
    input_width = input_width #or 90 for actitracker
    num_labels = 11  #or 6 for actitracker
    num_channels = 52 #or 3 for actitracker
elif dataset =="sph":
    print "sph"
    input_height = 1
    input_width = input_width #or 90 for actitracker
    num_labels = 20  #or 6 for actitracker
    num_channels = 52 #or 3 for actitracker
else:
    print "wrong dataset"


learning_rate = 0.001
training_iters = 100000
batch_size = 64
display_step = 10



# Network Parameters
n_input = num_channels # MNIST data input (img shape: 28*28)
n_steps = input_width # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = num_labels # MNIST total classes (0-9 digits)


# DEFINE MODEL

# tf Graph input
# n_steps == window_size ||||  n_input == features 
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def BiRNN(x, weights, biases):

    x = tf.unstack(x, n_steps, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.5)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.5)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = BiRNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

training_epochs = 50
loss_over_time = np.zeros(training_epochs)
total_batches = train_x.shape[0] // batch_size
b = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    # cost_history = np.empty(shape=[0],dtype=float)
    for epoch in range(training_epochs):
        cost_history = np.empty(shape=[0],dtype=float)
        for b in range(total_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :]
            batch_y = train_y[offset:(offset + batch_size), :]
            
            # print "batch_x shape =",batch_x.shape
            # print "batch_y shape =",batch_y.shape

            _, c = sess.run([optimizer, cost],feed_dict={x: batch_x, y : batch_y})
            cost_history = np.append(cost_history,c)
        loss_over_time[epoch] = np.mean(cost_history)
        print "Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",sess.run(accuracy, feed_dict={x: train_x, y: train_y})
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y})

    # MORE METRICS
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    # pred_Y is the result of the FF-RNN
    y_p = tf.argmax(pred, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:test_y})
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
    print "Confusion matrix"
    print metrics.confusion_matrix(y_true, y_pred)
    #fpr, tpr, tresholds = metrics.roc_curve(y_true, y_pred)
    # plt.figure(1)
    # plt.plot(loss_over_time)
    # plt.title("Loss value over epochs (BDLSTM DG)")
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

    # while step * batch_size < training_iters:
    #     offset = (step * batch_size) % (train_y.shape[0] - batch_size)
    #     batch_x = train_x[offset:(offset + batch_size), :, :]
    #     batch_y = train_y[offset:(offset + batch_size), :]
    #     # batch_x, batch_y = mnist.train.next_batch(batch_size)
    #     # Reshape data to get 28 seq of 28 elements
    #     batch_x = batch_x.reshape((batch_size, n_steps, n_input))
    #     # Run optimization op (backprop)
    #     sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    #     if step % display_step == 0:
    #         # Calculate batch accuracy
    #         acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    #         # Calculate batch loss
    #         loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
    #         print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
    #               "{:.6f}".format(loss) + ", Training Accuracy= " + \
    #               "{:.5f}".format(acc))
    #     step += 1
    # print("Optimization Finished!")

    # print "Testing Accuracy:", session.run(accuracy, feed_dict={x: test_x, y: test_y})
    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
print("--- %s seconds ---" % (time.time() - start_time))
print "done."