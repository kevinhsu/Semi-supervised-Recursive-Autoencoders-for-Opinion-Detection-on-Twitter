UCL -  Information Retrieval and Data Mining Group Project

Sentiment Analysis on Political Opinion
======================================================

Problem description
------------------------------------------------------
Twitter data from the first 2008 Presidential debate
Total number of tweets is 3,238

Data is in ./data folder:
- rawdata.tsv - the whole data set
- test.tsv - a small set for testing purpose
- trainset - the training data
- testset - the test data


Prerequisite
------------------------------------------------------
The system is buit on Python 2.7

Other packages you need to install before running the system:
- numpy
- scikit learn
- matplotlib
- nltk


How to use
------------------------------------------------------
1. Download the code from github:

   git clone https://github.com/liyazhe/IRDMProject.git

2. run the following depending on the usage

   - for cross validation of the training data, run:

   crossvalidation.py data/model.json

   - for analysis, run:

   analysis.py data/model.json

   - for training and predict test data, run:

   main.py data/model.json


Model configuration
------------------------------------------------------
Parameters of the model can be changed in ./data/model/json

- d : dimension of the word vector
- cat : number of categories of the classification problem
- alpha : the proportion of supervised (classification) error and unsupervised (reconstruction) error
- lambdaW : regularisation term on word vector reconstruction matrices
- lambdaCat : regularisation term on category
- lambdaL : regularisation term on word embedding
- iter : number of maximum iteration of the minFunc solver
