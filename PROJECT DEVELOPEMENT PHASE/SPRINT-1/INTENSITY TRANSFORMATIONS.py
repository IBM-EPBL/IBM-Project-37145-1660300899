#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


live = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d2062c0-076d-4b98-b9f4-681f9b728cd9___JR_FrgE.S 8633.JPG')
mask = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d0776a1-7aba-4923-93b5-7ec629ca108b___JR_FrgE.S 2902.JPG')

plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(live, cmap='gray')
plt.subplot(122), plt.imshow(mask, cmap='gray')
plt.show()


# In[3]:


plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(live, cmap='gray')
plt.subplot(122), plt.imshow(live - 20, cmap='gray')
plt.show()


# In[4]:


plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(mask - live, cmap='gray')
plt.subplot(132), plt.imshow(-(mask - live + 128), cmap='gray')
plt.subplot(133), plt.imshow(mask - live + 128, cmap='gray')
plt.show()


# In[6]:


shaded = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d2062c0-076d-4b98-b9f4-681f9b728cd9___JR_FrgE.S 8633.JPG')
shading = imread(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___Black_rot\7d0776a1-7aba-4923-93b5-7ec629ca108b___JR_FrgE.S 2902.JPG')

plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(shaded, cmap='gray')
plt.subplot(122), plt.imshow(shading, cmap='gray')
plt.show()


# In[ ]:




