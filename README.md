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
The motivation for the creation of this repository lies with my MSc thesis project.

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
<pre><i>python (executable) (three first letters of the dataset)  
e.g. python cnn1d.py dap
</i></pre>

...or you can retrieve the f1-score / f-weighted and f-mean score of your test-set by using the already trained models inside the
trainedModels folder by typing:
<pre><i>I HAVENT DONE THIS PART YET</i></pre>

<h3>Model and basic functions description</h3>

Write stuff here
