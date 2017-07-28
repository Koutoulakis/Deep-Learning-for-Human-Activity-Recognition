# Deep Learning for Human Activity Recognition

<h3>Synopsis</h3>
In this repository a collection of deep learning networks (such as Convolutional Neural Networks -CNNs or Covnets-, Deep Feed Forward Neural Networs, also known as Multilayer Perceprtons -DNNs or MLPs-, Recurrent Neural Networks -RNNs-, specifically two flavors called Feed Forward Long Short Term Memory RNNs -FFLSTM-RNNs- and Bi-Directional LSTM RNNs i.e. -BDLSTMs-) will be trained and tested in order to tackle the Human Activity Recognition (HAR) problem.
We use python 2.7 and tensorflow throughout this tutorial.

The goal is to provide the public with a sequence of functions and files that would allow them in turn, with minimum changes, train test and evaluate their own datasets with state of the art deep learning approaches.

<h3>Repository Structure</h3>

The main structure that we are going to follow is this:

|Datareader  
|---->datareader.py  
|ModelCreation  
|----->|CNN  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---->cnn1d.py  
|----->|DNN  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---->mlp.py  
|----->|RNN  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---->|FFLSTM  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---->fflstm.py  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---->|BDLSTM  
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---->bdlstm.py  
|TrainedModels  
|----->cnn  
|----->mlp  
|----->fflstm  
|----->bdlstm  

<h3>Motivation</h3>
The motivation for the creation of this repository lies in my MSc thesis project.

<h3>Installation</h3>

Below is a small description of the process the practitioner should follow in order to run the code provided or further extend it to suit his/her own purposes.  

<b>PREPROSSES DATASETS STAGE</b>

The Datareader folder contains a python script that follows the logic of this github repository : https://github.com/nhammerla/deepHAR  
Its main purpose is to read a dataset and save it in a special file format known as hdf5 (https://support.hdfgroup.org/HDF5/), so that
in our main code we could extract the training and testing set of every different dataset just by changing the path and not anything else in the code. More datasets can be added by implementing additional if conditions inside the datareader.py  .

The datasets that we are preprocessing are the following :  
<b>Opportunity</b> dataset (it can be downloaded from : https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)  
<b>Daphnet Gait</b> dataset (it can be downloaded from : https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait)  
<b>PAMAP2</b> dataset (it can be downloaded from : https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)  
<b>Sphere</b> dataset (it can be downloaded from : https://data.bris.ac.uk/data/dataset/8gccwpx47rav19vk8x4xapcog)  

The way to do that is by downloading the dataset from the links provided above, adding the datareader.py script inside their
folder and running the command in a terminal:  
<pre><i>python datareader.py (dataset-three-first-initial-letters)</i></pre>  
e.g. for Daphnet gait you would have to type:  
<pre><i>python datareader.py dap</i></pre>

After this stage is complete, a file named daphnet.h5 will have been created containing the training and testing sets that were decided to be used for this dataset in the datareader.py (Note: the training-test split we follow for the Opportunity, Daphnet and PAMAP2 datasets are described in Deep, Convolutional, and Recurrent Models for Human Activity Recognition using Wearables, by Nils Y. Hammerla, Shane Halloran, Thomas Ploetz (https://arxiv.org/abs/1604.08880). The split for the Sphere dataset is already created for us in its hosting site.)  
We can later on call the the dataset in our code by fixing the correct path in the line calling the os python function.
e.g. :  
<pre><i>path = os.path.join(os.path.expanduser('~'), 'Downloads', 'OpportunityUCIDataset', 'opportunity.h5')</i></pre>    
Note: This path function works both for windows and linux.
The correct dataset with its train-test split is retrieved by simply typing :
<pre><i>
f = h5py.File(path, 'r')
x_train = f.get('train').get('inputs')[()]
y_train = f.get('train').get('targets')[()]
x_test = f.get('test').get('inputs')[()]
y_test = f.get('test').get('targets')[()]
</i></pre>

At this point you can run the code contained inside the ModelCreation folder to train a model.
This can be done by going inside a specifc model's folder (e.g. CNN) and typing the following command :
<pre><i>python (executable) (three first letters of the dataset)</i></pre>  
e.g.In order to train Daphnet Gait with a CNN you would have to type:
<pre><i>python cnn1d.py dap</i></pre>

...or you can retrieve the f1-score / f-weighted and f-mean score of your test-set by using the already trained models inside the
trainedModels folder by typing:
<pre><i>I HAVENT DONE THIS PART YET</i></pre>

<h3>Models and basic functions description</h3>

<b>Models</b>
Initially, in all the following models except from the DNN-MLP, we use the datasets as is, and we split them in 1-5 second window segments as described in https://arxiv.org/abs/1604.08880 with 50% overlap. For the DNN, we flatten the input. E.g. if we have a dataset with 3 measurements x,y,z that contains 1000 samples and has a window size of 23 (ie 23 consecutive measurements, which, depending on the frequency of the measurements, it corresponds to 1-5 seconds of activity), the input dimension for the network will not be (1000,23,3) but it will be (1000,23*3) with the x,y,z having the formation x1,x2,..x23,y1,y2,..y23,z1,z2,...z23. 
<ol>
<li>Deep Feed Forward Neural Network - Multilayer Perceptron (DNN,MLP)</li>
<br>A short description of this network is given here: https://en.wikipedia.org/wiki/Feedforward_neural_network
The size of each layer depends on the dataset we use as an input, the smaller datasets contain fewer hidden layers, with fewer neurons each.<br>The goal was to replicate similar f-scores for the 3 datasets (Opportunity, Pamap2, Daphnet Gait) with the paper described here (https://arxiv.org/abs/1604.08880) so that we can examine the performance of these networks on the Sphere dataset and do some parameter and optimization function exploration.<br>
There is a an if-else section in each algorithm which changes the parameters depending on which dataset (dap,opp,pam,sph) is given. In case another dataset name or a name is not given, it returns an error code. This part of the code could be changed as to include an additional dataset with different parameter configurations.

<li>Convolutional Neural Network (CNN)</li>
<br>The network is described here : https://en.wikipedia.org/wiki/Convolutional_neural_network
<br>We follow the same logic described in the paper mentioned above.
After the training process both recall and precision is calculated. The confusion matrix is printed as well.
<li>Long Short Term Memory Recurrent Neural Network (LSTM-RNN)</li>
<br>The network is described here : https://en.wikipedia.org/wiki/Long_short-term_memory
<br>Same logic as above applies.
<li>Bi-Directional Long Short Term Memory Recurrent Neural Network (BD-LSTM-RNN)</li>
<br>The bi directional concept of an RNN is described here : https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks
<br>Same logic as above applies.
</ol>
