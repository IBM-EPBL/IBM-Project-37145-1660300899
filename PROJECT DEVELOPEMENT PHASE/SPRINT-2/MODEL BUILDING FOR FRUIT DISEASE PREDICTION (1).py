#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1)


# In[2]:


x_train=train_datagen.flow_from_directory(r'C:\Users\91638\Downloads\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\train',target_size=(128,128),batch_size=2,class_mode='categorical')
x_test=test_datagen.flow_from_directory(r'C:\Users\91638\Downloads\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test',target_size=(128,128),batch_size=2,class_mode='categorical')


# In[3]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[4]:


model=Sequential()


# In[5]:


model.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))


# In[6]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[7]:


model.add(Flatten())


# In[8]:


model.add(Dense(units=40,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=70,kernel_initializer='random_uniform',activation='relu'))
model.add(Dense(units=6,kernel_initializer='random_uniform',activation='softmax'))


# In[9]:


model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=["accuracy"])


# In[10]:


model.fit(x_train,steps_per_epoch=168,epochs=3,validation_data=x_test,validation_steps=52)


# In[11]:


model.save(r'C:\Users\91638\Downloads\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload\fruit.h5')


# In[12]:


model.summary()


# In[ ]:


model.save


# In[18]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[20]:


model=load_model(r'C:\Users\91638\Downloads\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload\fruit.h5')


# In[23]:


image=image.load_img(r"C:\Users\91638\Downloads\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\fruit-dataset\fruit-dataset\test\Apple___healthy\00fca0da-2db3-481b-b98a-9b67bb7b105c___RS_HL 7708.JPG")


# In[24]:


image


# In[25]:


model.summary()


# In[ ]:




