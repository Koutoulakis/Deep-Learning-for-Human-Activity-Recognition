# Deep Learning for Human Activity Recognition

In this repository a collection of deep learning networks will be trained and tested in order to tackle the Human Activity Recognition (HAR) problem.
We use python 2.7 and tensorflow throughout this tutorial.

The goal is to provide the public with a sequence of functions and files that would allow them in turn, with minimum changes, train test and evaluate their own datasets.

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

#PREPROSSES DATASETS STAGE

The Datareader folder contains a python script that follows the logic of this github repository : https://github.com/nhammerla/deepHAR  
Its main purpose is to read a dataset and save it in a special file format known as hdf5 (https://support.hdfgroup.org/HDF5/), so that
in our main code we could extract the training and testing set of every different dataset just by changing the path and not anything else in the code. More datasets can be added by implementing additional if conditions inside the datareader.py  .

The datasets that we are preprocessing are the following :  
Opportunity dataset (it can be downloaded from : https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)  
Daphnet Gait dataset (it can be downloaded from : https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait)  
PAMAP2 dataset (it can be downloaded from : https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)  
Sphere dataset (it can be downloaded from : https://data.bris.ac.uk/data/dataset/8gccwpx47rav19vk8x4xapcog)  

The way to do that is by writing downloading the dataset from the links provided above, adding the datareader.py script inside their
folder and running the command:
$python datareader.py <dataset-three-first-initial-letters>
e.g. for Daphnet gait you would have to type :$python datareader.py dap

After this stage is complete, 
