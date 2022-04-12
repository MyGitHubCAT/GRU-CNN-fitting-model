#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[53]:


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

#type数据
typemean=typearr.mean(axis=0)
typestd=typearr.std(axis=0)
normtype=(typearr-typemean)/typestd

#RATE数据
ratemean = ratearr.mean(axis=0)
ratestd = ratearr.std(axis=0)
normrate = ( ratearr-ratemean)/ratestd

#收益数据
aveRarr=np.array(normaldf['aveRevenue'])
aveGRarr=np.array(normaldf['aveGrossRevenue'])

#目标停车时长
labeltime = np.array(normaldf['time'])


# In[4]:


#加载训练好的模型
typeGRU = tf.keras.models.load_model(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\typeGRU')
rateGRU = tf.keras.models.load_model(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\rateGRU')
aveRGRU = tf.keras.models.load_model(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\aveRGRU')
aveGRGRU = tf.keras.models.load_model(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\rateGGRU')
fitmodel = tf.keras.models.load_model(r'D:\sjtu\博1\课程\交通运输经济学\savemodel\fitmodel')


# In[92]:


typeval = normtype[len(normtype)-28:]
rateval = normrate[len(normrate)-28:]
aveRval = aveRarr[len(aveRarr)-28:]
aveGRval = aveGRarr[len(aveGRarr)-28:]

labeltimeval = labeltime[len(labeltime)-28:]


# In[93]:


def prevalue(inps,model): #基于模型预测数据
    predata = []
    for i in range(len(inps)-7+1):
        minp = inps[slice(i,i+7)]
        minp = np.expand_dims(minp,axis=0)
        res = model(minp)
        predata.append(res)
    return np.concatenate(predata,axis=1)


# In[94]:


typepre=prevalue(typeval,typeGRU)
ratepre=prevalue(rateval,rateGRU)
aveRpre=prevalue(aveRval,aveRGRU)
aveGRpre=prevalue(aveGRval,aveGRGRU)

timeval = (np.transpose(aveRpre,(1,0)),np.transpose(aveGRpre,(1,0)),
           np.sum(ratepre,axis=0),np.sum(typepre,axis=0))


# In[95]:


pretime = fitmodel(timeval)
#fitmodel.input_shape


# In[96]:


#最后一次预测没有数据不能进行比较
prelabel =  pretime[:-1].numpy().sum(axis=1)
timeval = labeltimeval[7:]


# In[101]:


labels =(prelabel*dfstd['time'])+dfmean['time']
preres =(timeval*dfstd['time'])+dfmean['time']


# In[105]:


x = np.arange(21)
plt.figure(figsize=(24,12))
plt.plot(x,labels,label='actual parking time')
plt.plot(x,preres,label='prediction data' )
plt.legend(fontsize = 24)
plt.xlabel('date/days',fontsize = 24)
plt.ylabel('parking time',fontsize = 24)
plt.tick_params(labelsize=24)


# In[113]:


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


# In[114]:


CalMAE(labels,preres)


# In[115]:


CalRMSE(labels,preres)


# In[116]:


CalR2(labels,preres)


# In[ ]:




