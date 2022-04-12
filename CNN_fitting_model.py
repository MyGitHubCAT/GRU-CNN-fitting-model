#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#拟合模型


# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


datadf = pd.read_json(r'D:\sjtu\博1\课程\交通运输经济学\traindata.json')

#标准化处理
#注意独热数据需要单独分析
ohrate = datadf.pop('One-Hot Rate')
ohtype = datadf.pop('One-Hot Type') 

dfmean = datadf.mean()
dfstd = datadf.std()
normaldf = (datadf-dfmean)/dfstd


#独热数据需要单独分析
ohrarr = []
ohtarr = []
for i in ohrate.map(lambda x: np.array(x)):
    ohrarr.append(i)
for i in ohtype.map(lambda x: np.array(x)):
    ohtarr.append(i)
ratearr = np.array(ohrarr)
typearr = np.array(ohtarr)

typemean=typearr.mean(axis=0)
typestd=typearr.std(axis=0)
normtype=(typearr-typemean)/typestd

ratemean = ratearr.mean(axis=0)
ratestd = ratearr.std(axis=0)
normrate = ( ratearr-ratemean)/ratestd


# In[66]:


InputR = tf.keras.Input(shape=(1,))
InputGR = tf.keras.Input(shape=(1,))
InputRate = tf.keras.Input(shape=(9,))
InputType = tf.keras.Input(shape=(2,))
denseR = tf.keras.layers.Dense(12,activation = 'relu')(InputR)
denseGR = tf.keras.layers.Dense(12,activation = 'relu')(InputGR)
denseRate = tf.keras.layers.Dense(12,activation = 'relu')(InputRate)
denseType = tf.keras.layers.Dense(12,activation = 'relu')(InputType)
comf = tf.concat([denseR,denseGR,denseRate,denseType],axis=-1)
exdimf = tf.expand_dims(comf,axis=-1)
cnn1 = tf.keras.layers.Conv1D(16,3,2,activation = 'relu')(exdimf)
cnn2 = tf.keras.layers.Conv1D(32,3,2,activation = 'relu')(cnn1)
flat2 = tf.keras.layers.Flatten()(cnn2)
dense3 = tf.keras.layers.Dense(32)(flat2)
denseres = tf.keras.layers.Dense(1)(dense3)
model = tf.keras.Model(inputs=[InputR,InputGR,InputRate,InputType],outputs = denseres)


# In[67]:


model.summary()


# In[68]:


tf.keras.utils.plot_model(model, r"D:\sjtu\博1\课程\交通运输经济学\image\fittingmodel.png", show_shapes=True)


# In[69]:


dataR = np.expand_dims(np.array(normaldf['aveRevenue']),axis=-1)
dataGR= np.expand_dims(np.array(normaldf['aveGrossRevenue']),axis=-1)
dataRate = normrate
dataType = normtype
dataLabel = np.expand_dims(np.array(normaldf['time']),axis=-1)


# In[70]:


#输入数据集
dsR = tf.data.Dataset.from_tensor_slices(dataR ).batch(3)
dsGR = tf.data.Dataset.from_tensor_slices(dataGR ).batch(3)
dsRate = tf.data.Dataset.from_tensor_slices(dataRate ).batch(3)
dsType = tf.data.Dataset.from_tensor_slices(dataType ).batch(3)
#目标集
dsLabel= tf.data.Dataset.from_tensor_slices(dataLabel ).batch(3)


# In[71]:


modelds = tf.data.Dataset.zip((dsR ,dsGR,dsRate,dsType))
trainds = tf.data.Dataset.zip((modelds,dsLabel))


# In[72]:


model.compile(loss = tf.losses.MeanSquaredError(),
             metrics = [tf.metrics.MeanAbsoluteError()])
his = model.fit(trainds ,epochs=100)


# In[84]:


#保存模型
model.save(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\fitmodel')


# In[83]:


import matplotlib.pyplot as plt
x = np.arange(len(his.history['loss']))
plt.figure(figsize = (16,12))
plt.plot(x,his.history['loss'],label='train loss')
plt.xlabel('loss',fontsize = 24)
plt.ylabel('step',fontsize = 24)
plt.tick_params(labelsize=24)
plt.legend(fontsize =24)


# In[ ]:




