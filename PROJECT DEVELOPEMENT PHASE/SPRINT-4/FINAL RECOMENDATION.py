#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


fertilizer_data_path = r'C:\Users\91638\Downloads\archivee\fertilizer.csv'
merge_fert = pd.read_csv(fertilizer_data_path)


# In[3]:


merge_fert.head()


# In[8]:


del merge_fert['Potassium']


# In[9]:


merge_fert.describe()


# In[12]:


merge_fert['Crop Type'].unique()


# In[14]:


plt.plot(merge_fert["Nitrogen"])


# In[15]:


plt.plot(merge_fert["Moisture"])


# In[18]:


plt.plot(merge_fert["Phosphorous"])


# In[19]:


sns.heatmap(merge_fert.corr(),annot=True)


# In[25]:


merge_crop = pd.read_csv( r'C:\Users\91638\Downloads\archivee\crop.csv')
reco_fert = merge_fert


# In[ ]:





# In[ ]:




