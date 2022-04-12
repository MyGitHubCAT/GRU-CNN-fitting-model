#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#对type,ave等因素预测，再基于这几个因素分析每天的停车总时长


# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


datadf = pd.read_json(r'D:\sjtu\博1\课程\交通运输经济学\traindata.json')
datadf


# In[3]:


#标准化处理
#注意独热数据需要单独分析
ohrate = datadf.pop('One-Hot Rate')
ohtype = datadf.pop('One-Hot Type') 

dfmean = datadf.mean()
dfstd = datadf.std()
normaldf = (datadf-dfmean)/dfstd
normaldf


# In[15]:


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


def predictdata(data,res,isarr=True):    #基于输出与预测值绘制图像   
    x = np.arange(len(data)+1)
    
    if isarr:
        dimnums = data.shape[1]
        fig=plt.figure(figsize=(24,8*dimnums))        
        for i in range(dimnums):
            ax=fig.add_subplot(dimnums,1,i+1)
            ax.plot(x[:-1],data[:,i],label='observed data')
            ax.plot(x[7:None],res[:,i],label='prediction results')
            ax.legend(fontsize=28)
            ax.set_xlabel('timeindex/days',fontsize=28)
            ax.set_ylabel('result',fontsize=28)
            ax.tick_params(labelsize=28)
    else:
        fig=plt.figure(figsize=(24,8))                
        ax=fig.add_subplot(1,1,1)
        ax.plot(x[:-1],data,label='observed data')
        ax.plot(x[7:None],res,label='prediction results')
        ax.legend(fontsize=28)
        ax.set_xlabel('timeindex/days',fontsize=28)
        ax.set_ylabel('result',fontsize=28)
        ax.tick_params(labelsize=28)


# In[5]:


#独热数据需要单独分析
ohrarr = []
ohtarr = []
for i in ohrate.map(lambda x: np.array(x)):
    ohrarr.append(i)
for i in ohtype.map(lambda x: np.array(x)):
    ohtarr.append(i)
ratearr = np.array(ohrarr)
typearr = np.array(ohtarr)


# In[6]:


#type数据数据集建立与模型训练
typemean=typearr.mean(axis=0)
typestd=typearr.std(axis=0)
normtype=(typearr-typemean)/typestd
typedataset = tf.keras.utils.timeseries_dataset_from_array(normtype,targets=None,sequence_length=8,batch_size=3)
typeds = typedataset.map(split_data)

typeGRU = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7,2)),
    tf.keras.layers.GRU(64,activation='relu',return_sequences=True),
    tf.keras.layers.GRU(128,activation='relu',return_sequences=False),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1*2),
    tf.keras.layers.Reshape((1,2))
])
typeGRU.compile(loss = tf.losses.MeanSquaredError(),
                metrics = [tf.metrics.MeanAbsoluteError()])
typehistory = typeGRU.fit(typeds,epochs=50)


# In[16]:


#0.3数据的预测与绘图
typeinp = normtype[int(len(normtype)*0.7):]#预测最后0.3的数据
typepreres = prevalue(typeinp,typeGRU)
typepreres =typepreres.reshape((typepreres.shape[1],typepreres.shape[2]))
predictdata(typeinp,typepreres)  


# In[8]:


#rate数据数据集建立与模型训练
ratemean = ratearr.mean(axis=0)
ratestd = ratearr.std(axis=0)
normrate = ( ratearr-ratemean)/ratestd
#normrate
ratedataset = tf.keras.utils.timeseries_dataset_from_array(normrate,targets=None,sequence_length = 8,batch_size=3)
rateds = ratedataset.map(split_data)

rateGRU = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7,9)),
    tf.keras.layers.GRU(64,activation='relu',return_sequences=True),
    tf.keras.layers.GRU(128,activation='relu',return_sequences=False),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1*9),
    tf.keras.layers.Reshape((1,9))
])
rateGRU.compile(loss = tf.losses.MeanSquaredError(),
                metrics = [tf.metrics.MeanAbsoluteError()])
ratehistory = rateGRU.fit(rateds,epochs=50)


# In[17]:


rateinp = normrate[int(len(normrate)*0.7):]#预测最后0.3的数据
ratepreres = prevalue(rateinp,rateGRU)
ratepreres =ratepreres.reshape((ratepreres.shape[1],ratepreres.shape[2]))
predictdata(rateinp,ratepreres) 


# In[ ]:



# 


# In[10]:


#收益预测 建立数据集
aveRarr=np.array(normaldf['aveRevenue'])
aveGRarr=np.array(normaldf['aveGrossRevenue'])
aveRdataset = tf.keras.utils.timeseries_dataset_from_array(aveRarr,targets=None,sequence_length = 8,batch_size=3)
aveGRdataset = tf.keras.utils.timeseries_dataset_from_array(aveGRarr,targets=None,sequence_length = 8,batch_size=3)
aveRds = aveRdataset.map(split_data)
aveGRds = aveGRdataset.map(split_data)


# In[11]:


aveRGRU = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7,)),
    tf.keras.layers.Reshape(target_shape=(7,1)),
    tf.keras.layers.GRU(64,activation = 'relu',return_sequences=False),
    #tf.keras.layers.GRU(128,activation = 'relu',return_sequences=False),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1)
])

aveRGRU.compile(loss=tf.losses.MeanSquaredError(),
               metrics = [tf.metrics.MeanAbsoluteError()])
aveRhistory = aveRGRU.fit(aveRds,epochs=50)


# In[19]:


aveRinp = aveRarr[int(len(aveRarr)*0.7):]#预测最后0.3的数据
aveRpreres = prevalue(aveRinp,aveRGRU)
aveRpreres = aveRpreres.reshape((aveRpreres.shape[1],))
predictdata(aveRinp,aveRpreres,False)  
 


# In[13]:


aveGRGRU = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(7,)),
    tf.keras.layers.Reshape(target_shape=(7,1)),
    tf.keras.layers.GRU(64,activation = 'relu',return_sequences=True),
    tf.keras.layers.GRU(128,activation = 'relu',return_sequences=False),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1)
])

aveGRGRU.compile(loss=tf.losses.MeanSquaredError(),
               metrics = [tf.metrics.MeanAbsoluteError()])
aveGRhistory = aveGRGRU.fit(aveGRds,epochs=50)


# In[20]:


aveGRinp = aveGRarr[int(len(aveGRarr)*0.7):]#预测最后0.3的数据
aveGRpreres = prevalue(aveGRinp,aveGRGRU)
aveGRpreres = aveGRpreres.reshape((aveGRpreres.shape[1],))
predictdata(aveGRinp,aveGRpreres,False)  


# In[21]:


#保存模型
typeGRU.save(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\typeGRU')
rateGRU.save(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\rateGRU')
aveRGRU.save(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\aveRGRU')
aveGRGRU.save(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\rateGGRU')


# In[ ]:




