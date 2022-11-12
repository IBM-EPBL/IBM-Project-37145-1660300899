#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


# In[5]:


I = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d2062c0-076d-4b98-b9f4-681f9b728cd9___JR_FrgE.S 8633.JPG')
J = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d0776a1-7aba-4923-93b5-7ec629ca108b___JR_FrgE.S 2902.JPG')


# In[6]:


plt.figure()
plt.subplot(121), plt.imshow(I)
plt.subplot(122), plt.imshow(J)
plt.show()


# In[7]:


plt.figure(figsize=(10, 10))
plt.imshow(np.abs(I[:, :, 0].astype(float) - J[:, :, 0].astype(float)), cmap='gray')
plt.show()


# In[9]:


d = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d2062c0-076d-4b98-b9f4-681f9b728cd9___JR_FrgE.S 8633.JPG')
mask = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d0776a1-7aba-4923-93b5-7ec629ca108b___JR_FrgE.S 2902.JPG')

print(np.amin(d), np.amax(d))
print(np.amin(mask), np.amax(mask))


# In[10]:


plt.figure(), plt.imshow(mask), plt.show()


# In[11]:


mask = mask[:, :, 0]


# In[12]:


maskInv = np.zeros_like(mask)
maskInv[mask == 0] = 255
maskInv[mask == 255] = 0
plt.figure(), plt.imshow(maskInv, cmap='gray'), plt.show()


# In[ ]:




