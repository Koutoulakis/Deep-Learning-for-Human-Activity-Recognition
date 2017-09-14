import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import time
from sklearn import metrics
import h5py
import os
import sys
# import tflearn

#%matplotlib inline
plt.style.use('ggplot')



from random import shuffle

#shuffles two related lists #TODO check that the two lists have same size
# def shuffle_examples(examples, labels):
#     examples_shuffled = []
#     labels_shuffled = []
#     indexes = list(range(len(examples)))
#     shuffle(indexes)
#     for i in indexes:
#         examples_shuffled.append(examples[i])
#         labels_shuffled.append(labels[i])
#     examples_shuffled = np.asarray(examples_shuffled)
#     labels_shuffled = np.asarray(labels_shuffled)
#     return examples_shuffled, labels_shuffled
# FUNCTION DECLARATION

# def feature_normalize(dataset):
#     mu = np.mean(dataset,axis = 0)
#     sigma = np.std(dataset,axis = 0)
#     return (dataset - mu)/sigma
# def variable_summaries(var,name):
#   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#   with tf.name_scope('summaries_'+name):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)

def windowz(data, size):
    start = 0
    while start < len(data):
        yield start, start + size
        start += (size / 2)

def segment(x_train,y_train,window_size):
    # 9 is the number of features. we put window_size samples in the same line
    # with 9 * window_size per row. The format is, if we had 3 features, xyz
    # x1,y1,z1,x2,y2,z2,x3,y3,z3 etc, per row. 1->2->3...->window_size
    shapeX = x_train.shape
    segments = np.zeros(((len(x_train)//(window_size//2))-1,window_size*shapeX[1]))
    labels = np.zeros(((len(y_train)//(window_size//2))-1))
    i_segment = 0
    i_label = 0

    for (start,end) in windowz(x_train,window_size):
        if(len(x_train[start:end]) == window_size):
            m = stats.mode(y_train[start:end])
            offset_st=0
            offset_fin=window_size
            for i in range(shapeX[1]):
                segments[i_segment][offset_st:offset_fin] = (x_train[start:end])[:, [i]].T
                offset_st = (i+1)*window_size
                offset_fin = (i+2)*window_size
            # for i in range(window_size):
            #     segments[i_segment][offset_st:offset_fin] = x_train[start+i]
            #     offset_st+=9
            #     offset_fin+=9
            labels[i_label] = m[0]
            i_label+=1
            i_segment+=1
    return segments, labels

# MAIN ()


print "starting..."
start_time = time.time()

# DATA PREPROCESSING
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
    train_x, train_y = segment(x_train,y_train,input_width)
    test_x, test_y = segment(x_test,y_test,input_width)
    print "signal segmented."
elif dataset =="dap":
    print "dap seg"
    input_width = 25
    print "segmenting signal..."
    train_x, train_y = segment(x_train,y_train,input_width)
    test_x, test_y = segment(x_test,y_test,input_width)
    print "signal segmented."
elif dataset =="pa2":
    print "pa2 seg"
    input_width = 25
    print "segmenting signal..."
    train_x, train_y = segment(x_train,y_train,input_width)
    test_x, test_y = segment(x_test,y_test,input_width)
    print "signal segmented."
elif dataset =="sph":
    print "sph seg"
    input_width = 25
    print "segmenting signal..."
    train_x, train_y = segment(x_train,y_train,input_width)
    test_x, test_y = segment(x_test,y_test,input_width)
    print "signal segmented."
else:
    print "no correct dataset"

print "train_x shape =",train_x.shape
print "train_y shape =",train_y.shape
print "test_x shape =",test_x.shape
print "test_y shape =",test_y.shape

# train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
# test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)


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

# Parameters
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

learning_rate = 0.001#1e-3
training_epochs = 500
batch_size = 64
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_hidden_3 = 256 # 3nd layer number of features
n_hidden_4 = 256 # 4nd layer number of features
# n_input = 90 # MNIST data input (img shape: 28*28)
# n_classes = 6 # MNIST total classes (0-9 digits)
dropout1 = tf.placeholder(tf.float64) #0.5
dropout2 = tf.placeholder(tf.float64)
# input_height = 1
# input_width = 90
# num_labels = 6  #ie walking,running, etc
# num_channels = 3 #x,y,z, ie the features 



# tf Graph input
# x = tf.placeholder("float", [None, n_input])
x = tf.placeholder(tf.float64, shape=[None,input_height*input_width*num_channels])
y = tf.placeholder(tf.float64, shape=[None,num_labels])
# y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases,dropout1=1-0.5,dropout2=1-0.5):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.dropout(layer_1, dropout1)
    layer_1 = tf.nn.relu(layer_1)


    # variable_summaries(weights['h1'],'W_h1')
    # variable_summaries(biases['b1'],'b_b1')
    # tf.summary.histogram('layer_1', layer_1)
    # tf.summary.histogram('layer_1/sparsity', tf.nn.zero_fraction(layer_1))

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.dropout(layer_2, dropout2)
    layer_2 = tf.nn.relu(layer_2)

    # variable_summaries(weights['h2'],'W_h2')
    # variable_summaries(biases['b2'],'b_b2')
    # tf.summary.histogram('layer_2', layer_2)
    # tf.summary.histogram('layer_2/sparsity', tf.nn.zero_fraction(layer_2))

    # Hidden layer with RELU activation
    # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.dropout(layer_3, dropout2)
    # layer_3 = tf.nn.relu(layer_3)

    # tf.summary.histogram('layer_3', layer_3)
    # tf.summary.histogram('layer_3/sparsity', tf.nn.zero_fraction(layer_3))

    # Hidden layer with RELU activation
    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)
    # layer_4 = tf.nn.dropout(layer_4, dropout)

    # tf.summary.histogram('layer_4', layer_4)
    # tf.summary.histogram('layer_4/sparsity', tf.nn.zero_fraction(layer_4))

    # Output layer with linear activation
    # out_layer = tf.add(tf.matmul(layer_3, weights['out']),biases['out'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['out']),biases['out']))
    # variable_summaries(weights['out'],'W_out')
    # variable_summaries(biases['out'],'b_out')
    # tf.summary.histogram('out_layer', out_layer)
    # tf.summary.histogram('out_layer/sparsity', tf.nn.zero_fraction(out_layer))

    # out_layer = tf.Print(out_layer, [tf.argmax(out_layer, 1)],'argmax(out) = ', summarize=50, first_n=50)
    return out_layer

# Store layers weight & bias
weights = {
    # h1 shape : [1,input_width, num_channels, n_hidden_1]   --> [n_input, n_hidden_1]
    'h1': tf.Variable(tf.random_normal([num_channels*input_width,n_hidden_1], stddev=0.1, dtype=np.float64)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1, dtype=np.float64)),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=0.1, dtype=np.float64)),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=0.1, dtype=np.float64)),
    'out': tf.Variable(tf.random_normal([n_hidden_4, num_labels], stddev=0.1, dtype=np.float64))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=np.float64)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=np.float64)),
    'b3': tf.Variable(tf.random_normal([n_hidden_3], dtype=np.float64)),
    'b4': tf.Variable(tf.random_normal([n_hidden_4], dtype=np.float64)),
    'out': tf.Variable(tf.random_normal([num_labels], dtype=np.float64))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)



# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)) 

# tf.summary.scalar('cost', cost)
# variable_summaries(cost,'cost')
# global_step = tf.Variable(0, dtype=tf.int32, trainable=False)


# Create an optimizer.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# # Compute the gradients for a list of variables.
# grads_and_vars = optimizer.compute_gradients(cost, [weights['h1'],weights['h2'],weights['out']])
# # grads_and_vars is a list of tuples (gradient, variable).
# # Do whatever you need to the 'gradient' part, for example cap them, etc.
# capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=3.0, axes=0), gv[1])
#                          for gv in grads_and_vars]
# # Ask the optimizer to apply the capped gradients
# optimizer = optimizer.apply_gradients(capped_grads_and_vars)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
# Initializing the variables
# init = tf.global_variables_initializer()


# for v in tf.trainable_variables():
#      tf.summary.histogram(v.name, v)


total_batches = train_x.shape[0] // batch_size
print "total_batches=",total_batches
loss_over_time = np.zeros(training_epochs)
# Launch the graph
with tf.Session() as sess:
    # sess.run(init)
    # merged_summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter("./", sess.graph)
    tf.initialize_all_variables().run()
    # Training cycle
    for epoch in range(training_epochs):
        # merged_summary_op = tf.constant(1)
        # merged_summary_op = tf.summary.merge_all() #<---use this not the above one
        # summary_writer = tf.summary.FileWriter("./", sess.graph)
        avg_cost = 0.
        total_batch = total_batches #int(mnist.train.num_examples/batch_size)
        # train_x, train_y = shuffle_examples(train_x, train_y)
        # Loop over all batches
        cost_history = np.empty(shape=[0],dtype=float)
        for i in range(total_batch):

            # batch_x = train_x[i*batch_size:(i+1)*batch_size]
            # batch_y = train_y[i*batch_size:(i+1)*batch_size]

            offset = (i * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :]
            batch_y = train_y[offset:(offset + batch_size), :]

            # print "batch_x.shape=",batch_x.shape
            # print "batch_y.shape=",batch_y.shape
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            # _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x,
            #                                               y: batch_y, dropout1 : 1 - 0.3,dropout2:1-0.5})
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y, dropout1 : 1 - 0.5,dropout2:1-0.5})
            
            # if i>=37 and i<=41:
            #     print "i=",i,"c=",c
            #     print "batch_x.shape=",batch_x.shape
            #     print "batch_y.shape=",batch_y.shape
            #     print "weights[h1]",weights["h1"]
            #     print "weights[h2]",weights["h2"]
            #     print "weights[out]",weights["out"]

            # print "i=",i,"c=",c
            cost_history = np.append(cost_history,c)
        loss_over_time[epoch] = np.mean(cost_history)
            # summary_writer.add_summary(summary,global_step.eval(session=sess))
            # summary_writer.add_summary(summary,global_step.eval(session=sess))
        # Display logs per epoch step
        # with f1 score
        # y_p = tf.argmax(pred, 1)
        # val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:test_y, dropout1 : 1,dropout2 : 1})
        # y_true = np.argmax(test_y,1)
        # print "Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",sess.run(accuracy, feed_dict={x: train_x, y: train_y, dropout1 : 1,dropout2:1}),"f1_score", metrics.f1_score(y_true, y_pred, average="macro")
        # Without f1 score
        print "Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",sess.run(accuracy, feed_dict={x: train_x, y: train_y, dropout1 : 1,dropout2:1})
    print("Optimization Finished!")
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y: test_y, dropout1 : 1,dropout2 : 1})
    y_p = tf.argmax(pred, 1)
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_x, y:test_y, dropout1 : 1,dropout2 : 1})
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

    # plt.figure(1)
    # plt.plot(loss_over_time)
    # plt.title("Loss value over epochs (DNN DG)")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

    # print "f1_score_micro", metrics.f1_score(y_true, y_pred, average="micro")
    # print "f1_score_macro", metrics.f1_score(y_true, y_pred, average="macro")
    # print "f1_score_weighted", metrics.f1_score(y_true, y_pred, average="weighted")
    # if dataset=="dap":
    #     print "f1_score_", metrics.f1_score(y_true, y_pred)
    #print "f1_score_samples", metrics.f1_score(y_true, y_pred, average="samples")
    print "confusion_matrix"
    print metrics.confusion_matrix(y_true, y_pred)
    #fpr, tpr, tresholds = metrics.roc_curve(y_true, y_pred)

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