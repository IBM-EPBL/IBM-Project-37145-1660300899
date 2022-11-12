#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1)


# In[2]:


x_train=train_datagen.flow_from_directory(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\Veg-dataset\Veg-dataset\train_set',target_size=(128,128),batch_size=2,class_mode='categorical')
x_test=test_datagen.flow_from_directory(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\Veg-dataset\Veg-dataset\test_set',target_size=(128,128),batch_size=2,class_mode='categorical')


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


model.add(Dense(units=300,kernel_initializer='uniform',activation='relu'))


# In[9]:


model.add(Dense(units=150,kernel_initializer='uniform',activation='relu'))


# In[10]:


model.add(Dense(units=75,kernel_initializer='uniform',activation='relu'))


# In[11]:


model.add(Dense(units=9,kernel_initializer='uniform',activation='softmax'))


# In[12]:


model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=["accuracy"])


# In[13]:


model.fit(x_train,steps_per_epoch=89,epochs=20,validation_data=x_test,validation_steps=27)


# In[14]:


model.save(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload2\veg.h5') 


# In[15]:


model.summary()


# In[19]:


model.save(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload2\veg.h5')


# In[17]:


import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[20]:


model=load_model(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\upload2\veg.h5')


# In[22]:


img=image.load_img(r'C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\Veg-dataset\Veg-dataset\train_set\Tomato___Bacterial_spot\1fbc778e-9d12-4813-bba5-75a7275ab525___GCREC_Bact.Sp 5983.JPG')


# In[23]:


img


# In[24]:


x=image.img_to_array(img)


# In[25]:


img=image.load_img(r"C:\Users\91638\OneDrive\Documents\HTML\Desktop\logesh\Fertilizers_Recommendation_ System_For_Disease_ Prediction\Dataset Plant Disease\Veg-dataset\Veg-dataset\train_set\Tomato___Bacterial_spot\1fbc778e-9d12-4813-bba5-75a7275ab525___GCREC_Bact.Sp 5983.JPG",target_size=(128,128))


# In[26]:


x


# In[27]:


x=np.expand_dims(x,axis=0)


# In[28]:


x


# In[31]:


x_train.class_indices


# In[ ]:





# In[ ]:




