#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


crop_data_path = r'C:\Users\91638\Downloads\archivee\crop.csv'
fertilizer_data_path = r'C:\Users\91638\Downloads\archivee/Fertilizer.csv'

crop = pd.read_csv(crop_data_path)
fert = pd.read_csv(fertilizer_data_path)


# In[4]:


crop.head()


# In[5]:


fert.head()


# In[6]:


def change_case(i):
    i = i.replace(" ", "")
    i = i.lower()
    return i


# In[13]:


crop.head()


# In[14]:


crop.tail()


# In[15]:


crop_names = crop['label'].unique()
crop_names


# In[16]:


fert.head


# In[26]:


f


# In[ ]:





# In[ ]:




