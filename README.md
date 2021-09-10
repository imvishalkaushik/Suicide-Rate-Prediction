# Suicide-Rate-Prediction
Machine Learning algorithms to predict suicide rates on analyzing and finding signals correlated to increased suicide rates among different countries globally.
#DATA SET
In our problem, the data that should be feeded for the machine to decide and predict effectively has to be measure of variability in depressive symptoms along with other relevant factors such as year, age, population, GDP for year, GDP per capita, generation etc.
# IMOPORT THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
# IMPORT THE DATSET
Our model follows Supervised Learning, which consists in learning the link between two datasets: the observed data X and an external variable y that we are trying to predict, usually called “target”.
# CHECKING FOR OUTLIERS
Outliers can mislead the training process of machine learning algorithms takes longer training times, less accurate and poor results.
# INVESTIGATING CORRELATION AND VISUALIZING WITH HEATMAP
Correlation measures the strength or degree of relationship between two variables, for example age, sex and number of suicides.
# DATA PREPROCESSING
This preprocessing includes:-
Data Cleaning
Data Transformation
Data Reduction
# SPLITTING THE DATASET
As we work with datasets, a machine learning algorithm works in two stages. We have split the data around 20%-80% between testing and training stages.
# TRYING LINEAR REGRESSION
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x).
# PERFORMANCE EVALUATION
Using Linear Regression getting Accuracy/Score:-
Training Score - 55.23% ,
Test Score - 53.50%
# TRYING RIDGE REGRESSION
Ridge Regression is a machine learning algorithm based on supervised learning. It is a regularization technique. It shrinks the parameters.
# PERFORMANCE EVALUATION
Using Ridge Regression getting Accuracy/Score:-
Training Score - 55.27% ,
Test Score - 53.52%
# TRYING DECISION TREE
Decision tree is a machine learning algorithm based on supervised learning. It uses tree representation to solve the problem in which each leaf node represent the class label and attributes are represented on the internal node of the tree. Decisin Tree uses two different types for selection of root node(attribute selection) :- ID3 & CART.
# PERFORMANCE EVALUATION
Using Decision Tree getting Accuracy/Score:-
Training Score - 100% ,
Test Score - 98.21%
# TRYING RANDOM FOREST REGRESSOR
Random forests or random decision forests are an ensemble learning method for classification & regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees’ habit of overfitting to their training set.
# PERFORMANCE EVALUATION
Using Decision Tree getting Accuracy/Score:-
Training Score - 99.88% ,
Test Score - 99.22%
# USING RANDOMIZED SEARCHCV
Randomized searchcv is a technique for evaluating a machine learning model & testing its performance. CV is commonly used in applied Machine Learning tasks. Here we are using Randomized Searchcv for finding better parameters of Random Forest Regressor that increase our model accuracy.
