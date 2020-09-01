# Ionosphere-signals-prediction
This project is about analyzing Ionosphere data and measuring the accuracies of the electromagnetic signal data. The radar statistics were gathered by an arrangement in Goose Bay, Labrador. This system involves a phased array of 16 high-frequency transmitters with an aggregate transferred power on the order of 6.4 kilowatts. Expected waves were handled by exercising an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the Goose Bay system. Two attributes per pulse number describe instances in this database. 

This dataset describes high-frequency antenna returns from high energy particles in the atmosphere, and whether the return shows structure or not. The problem is a binary classification that contains 351 instances and 35 numerical attributes. The majority of the data in this set are continuous data points which range between -1 and 1, with one binomial variable which defines the type of the electromagnetic signals.

The objective of the project is to measure the accuracies of ‘good’ instances and ‘bad’ cases by feeding the dataset to the machine learning models mentioned below and report some of the measures to improve the overall performance of the models. Predicting the good and bad signals is very important as these signals propagate through distant places and contribute in providing better communication and help in improving the navigation. 

We will predict the good and bad signal results using 3 methods - KNN, GLM and decision tree and then use ensemble techniques to improve the accuracy of the model. In the ensemble technique, we will use the stacking method. We observed that generalized linear model has better classification rate among the rest and after implementing stacking technique we were able to improve the overall performance of the stacked models. 

# Introduction 
Source Information: 
-- Donor: Vince Sigillito (vgs@aplcen.apl.jhu.edu) 
-- Date: 1989 
-- Source: Space Physics Group, Applied Physics Laboratory, Johns Hopkins University, MD 20723 

The first 34 columns are continuous numerical data which represent 17 pulse numbers of received electromagnetic signals. There are two attributes per pulse number, which is the time of the pulse and the pulse number. The 35th column is categorical data "good" or "bad". "good" means those radar showing evidence of some type of structure in the ionosphere. “bad" implies those radar does not indicate their signals pass through the ionosphere. 

# Implementation of the Project 

First, we install the necessary packages and load the required libraries as mentioned below and then we read the dataset in R. We convert the last column label feature from character to factor. Next, to identify the important features we applied fitted Boruta model with the data and found out that column two i.e, V2 is not important and therefore, we removed V2 from the dataset and Created the significant dataset with important variables only. Then we split the dataset to train dataset and test dataset. Once, we have the training and test datasets we made use of knn() available in Class library for implementing KNN algorithm and glm() to implement logistic regression and rpart () to implement decision tree methods on our dataset. We chose these methods for our prediction and data analysis as we have binomial variable data with a binomial output. Because the above-mentioned algorithms perform better while dealing with categorical data points, we decided to implement the aforesaid classification methods. After completing with our modelling, we decided to improve the resulted accuracies of the models by implementing ensemble technique and we chose stacking for this case because it’s designed to combine model outputs of different types.
