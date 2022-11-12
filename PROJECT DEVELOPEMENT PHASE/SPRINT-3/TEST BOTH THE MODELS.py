#!/usr/bin/env python
# coding: utf-8

# In[1]:


test_dir=r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\Veg-dataset\Veg-dataset\train_set'


# In[2]:


import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[10]:


model = tf.keras.models.load_model(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload2\veg.h5')


# In[8]:


test_datagen_1=ImageDataGenerator(rescale=1)
test_generator_1=test_datagen_1.flow_from_directory(test_dir,target_size=(128,128),batch_size=20,class_mode='categorical')


# In[9]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[12]:


model =load_model(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload\fruit.h5')


# In[19]:


img=image.load_img(r"C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\Veg-dataset\Veg-dataset\test_set\Tomato___Leaf_Mold\abff37d9-e870-4274-a035-923c2bf5edaf___Crnl_L.Mold 6626.JPG",target_size=(128,128))


# In[22]:


x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
y=np.argmax(model.predict(x),axis=1)
index=['Apple___Black_rot', 'Apple___healthy', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Peach___Bacterial_spot', 'Peach___healthy']
index[y[0]]


# In[27]:


x


# In[ ]:




