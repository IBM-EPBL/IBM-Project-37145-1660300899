#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv(r'C:\Users\91638\Downloads\archive\crop_recommendation.csv')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.size


# In[7]:


df.shape


# In[9]:


df.columns


# In[10]:


df['label'].unique()


# In[11]:


df.dtypes


# In[12]:


df['label'].value_counts()


# In[13]:


sns.heatmap(df.corr(),annot=True)


# In[14]:


features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = df['label']


# In[15]:


acc = []
model = []


# In[16]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[17]:


from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))


# In[18]:


from sklearn.model_selection import cross_val_score


# In[19]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)


# In[ ]:





# In[ ]:




