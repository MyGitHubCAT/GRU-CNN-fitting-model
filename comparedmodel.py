#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[4]:


import math
def CalMAE(arr1,comparr2):#MAE
    nums = len(arr1)
    if(len(comparr2)!=nums):
        return 0  
    dis = 0
    for i in range(nums):
        dis =dis+ abs(arr1[i]-comparr2[i])
    mae = dis/nums
    return mae

def CalRMSE(arr1,comparr2):#MSE
    nums = len(arr1)
    if(len(comparr2)!=nums):
        return 0  
    dis = 0
    for i in range(nums):        
        dis =dis+ math.pow(arr1[i]-comparr2[i],2)
    mse = dis/nums
    
    return math.sqrt(mse)


def CalR2(arr1,comparr2):#求R2
    nums = len(arr1)
    if(len(comparr2)!=nums):
        return 1
    arr1mean = arr1.mean()
    dis1=0
    dis2=0
    for i in range(nums):        
        dis1 = dis1+math.pow(arr1[i]-comparr2[i],2)
        dis2 = dis2+math.pow(arr1[i]-arr1mean,2)
    r2 = 1-(dis1/dis2)
    return r2


def prevalue(inps,model): #基于模型预测数据
    predata = []
    for i in range(len(inps)-7+1):
        minp = inps[slice(i,i+7)]
        minp = np.expand_dims(minp,axis=0)
        res = model(minp)
        predata.append(res)
    return np.concatenate(predata,axis=1)


def split_data(inpsdata):#数据划分，基于七天的数据预测下一天
    inp = inpsdata[:,slice(0,7)]
    label = inpsdata[:,slice(7,None)]
    return inp,label


# In[5]:


datadf = pd.read_json(r'D:\sjtu\博1\课程\交通运输经济学\traindata.json')

#标准化处理
#注意独热数据需要单独分析
ohrate = datadf.pop('One-Hot Rate')
ohtype = datadf.pop('One-Hot Type') 

dfmean = datadf.mean()
dfstd = datadf.std()
normaldf = (datadf-dfmean)/dfstd


#目标停车时长
labeltime = np.array(normaldf['time'])
timedataset = tf.keras.utils.timeseries_dataset_from_array(labeltime,targets=None,sequence_length = 8,batch_size=3)
timeds = timedataset.map(split_data)


# In[7]:


next(iter(timeds))


# In[8]:


CNNmodel = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7,)),
    tf.keras.layers.Reshape(target_shape=(7,1)),
    tf.keras.layers.Conv1D(32,2,1,activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(1)
])

CNNmodel.compile(loss=tf.losses.MeanSquaredError(),
               metrics = [tf.metrics.MeanAbsoluteError()])
CNNmodelhistory = CNNmodel.fit(timeds,epochs=50)


# In[9]:


GRUmodel = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7,)),
    tf.keras.layers.Reshape(target_shape=(7,1)),
    tf.keras.layers.GRU(64,activation = 'relu',return_sequences=False),
    #tf.keras.layers.GRU(128,activation = 'relu',return_sequences=False),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1)
])
GRUmodel.compile(loss=tf.losses.MeanSquaredError(),
               metrics = [tf.metrics.MeanAbsoluteError()])
GRUmodelhistory = GRUmodel.fit(timeds,epochs=50)


# In[28]:


labeltimeval = labeltime[len(labeltime)-28:]
timeval =  labeltimeval[7:]


# In[29]:


CNNpre = prevalue(labeltimeval,CNNmodel)
GRUpre = prevalue(labeltimeval,GRUmodel)
CNNres = CNNpre.sum(axis=0)[:-1]
GRUres = GRUpre.sum(axis=0)[:-1]

labels = (timeval*dfstd['time'])+dfmean['time']
CNNres = (CNNres*dfstd['time'])+dfmean['time']
GRUres  = (GRUres *dfstd['time'])+dfmean['time']


# In[30]:


x = np.arange(21)
plt.figure(figsize=(24,12))
plt.plot(x,labels,label='actual parking time')
plt.plot(x,CNNres,label='CNN data' )
plt.plot(x,GRUres,label='GRU data' )
plt.legend(fontsize = 24)
plt.xlabel('date/days',fontsize = 24)
plt.ylabel('parking time',fontsize = 24)
plt.tick_params(labelsize=24)


# In[32]:


print(CalMAE(labels,CNNres))
print(CalRMSE(labels,CNNres))
print(CalR2(labels,CNNres))


# In[33]:


print(CalMAE(labels,GRUres))
print(CalRMSE(labels,GRUres))
print(CalR2(labels,GRUres))


# In[39]:


baseres = np.ones_like(CNNres)
baseres=np.array([ x*labels.mean() for x in baseres ])
baseres


# In[40]:


print(CalMAE(labels,baseres))
print(CalRMSE(labels,baseres))
print(CalR2(labels,baseres))


# In[ ]:




