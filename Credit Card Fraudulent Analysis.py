#!/usr/bin/env python
# coding: utf-8

# Credit Card Kaggle Anamoly Detection

# Context:    
# It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

# Content:
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# Importing the Dependencies

# In[105]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[106]:


# loading the dataset to a Pandas DataFrame
creditcard_data = pd.read_csv("C:/Users/dell/Downloads/Credit Card Fraudulent/creditcard.csv")


# In[107]:


# first 5 rows of the dataset
creditcard_data.head()


# In[108]:


creditcard_data.tail()


# In[109]:


# dataset informations
creditcard_data.info()


# Exploratory Data Analysis

# In[110]:


count_classes = pd.value_counts(creditcard_data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[142]:


## Get the Fraud and the normal dataset 

Fraud = creditcard_data[creditcard_data['Class']==1]

normal = creditcard_data[creditcard_data['Class']==0]


# In[143]:


print(fraud.shape,normal.shape)


# In[144]:


## We need to analyze more amount of information from the transaction data
#How different are the amount of money used in different transaction classes?
fraud.Amount.describe()


# In[145]:


normal.Amount.describe()


# In[146]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[148]:


# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(Fraud.Time, Fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[8]:


# checking the number of missing values in each column
creditcard_data.isnull().sum()


# In[150]:


## Take some sample of the data

data1= creditcard_data.sample(frac = 0.1,random_state=1)

data1.shape


# In[153]:


creditcard_data.shape


# In[9]:


# distribution of legit transactions & fraudulent transactions
creditcard_data['Class'].value_counts()


# In[ ]:


#Determine the number of fraud and valid transactions in the dataset

Fraud = data1[data1['Class']==1]

Valid = data1[data1['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))


# In[158]:


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(creditcard_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# This Dataset is highly unblanced

# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction

# In[159]:


# separating the data for analysis
legit = creditcard_data[creditcard_data.Class == 0]
fraud = creditcard_data[creditcard_data.Class == 1]


# In[160]:


print(legit.shape)
print(fraud.shape)


# In[161]:


# statistical measures of the data
legit.Amount.describe()


# In[162]:


fraud.Amount.describe()


# In[163]:


# compare the values for both transactions
creditcard_data.groupby('Class').mean()


# Under-Sampling

# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

# Number of Fraudulent Transactions --> 492

# In[164]:


legit_sample = legit.sample(n=492)


# Concatenating two DataFrames

# In[165]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[166]:


new_dataset.head()


# In[167]:


new_dataset.tail()


# In[168]:


new_dataset['Class'].value_counts()


# In[169]:


new_dataset.groupby('Class').mean()


# Splitting the data into Features & Targets

# In[170]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[171]:


print(X)


# In[172]:


print(Y)


# Split the data into Training data & Testing Data

# In[173]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[174]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[175]:


model = LogisticRegression()


# In[176]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# Model Evaluation

# Accuracy Score

# In[177]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[178]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[179]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[180]:


print('Accuracy score on Test Data : ', test_data_accuracy)

