#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# In[2]:


datapath = 'D:\sjtu\博1\课程\交通运输经济学\(Smarking) Dataset.csv'
smarkingdata = pd.read_csv(datapath)


# In[3]:


smarkingdata


# In[4]:


def recovertime(timestr):
    splitt = timestr.split('T')
    retimestr = splitt[0]+' '+splitt[1]
    return retimestr


# In[5]:


smarkingdata['ENTRY']=smarkingdata['Entry'].apply(recovertime)
smarkingdata['EXIT']=smarkingdata['Exit'].apply(recovertime)


# In[6]:


len(smarkingdata.groupby('Rate').sum())
dicta = {'Daily':np.array([1,0,0,0,0,0,0,0,0],dtype='int'),'Evening':np.array([0,1,0,0,0,0,0,0,0],dtype='int'),'Event':np.array([0,0,1,0,0,0,0,0,0],dtype='int'),
'Hotel Rate':np.array([0,0,0,1,0,0,0,0,0],dtype='int'),'Multi-Day':np.array([0,0,0,0,1,0,0,0,0],dtype='int'),'Show Rate':np.array([0,0,0,0,0,1,0,0,0],dtype='int'),
'Underground Rate':np.array([0,0,0,0,0,0,1,0,0],dtype='int'),'Weekday':np.array([0,0,0,0,0,0,0,1,0],dtype='int'),'Weekend':np.array([0,0,0,0,0,0,0,0,1],dtype='int')}
smarkingdata['One-Hot Rate'] = smarkingdata['Rate'].map(lambda x: dicta[x])
smarkingdata


# In[7]:


smarkingdata.groupby('Type').sum()
dictb = {'Reserved':np.array([1,0],dtype='int'),'Walk In':np.array([0,1],dtype='int')}
smarkingdata['One-Hot Type'] = smarkingdata['Type'].map(lambda x: dictb[x])
smarkingdata


# In[8]:


alldaysnum = 0
for i in np.arange(len(smarkingdata)): 
    SAMKDAY=datetime.datetime.strptime(smarkingdata.loc[i]['EXIT'].split(' ')[0], "%Y-%m-%d")- datetime.datetime.strptime(smarkingdata.loc[i]['ENTRY'].split(' ')[0], "%Y-%m-%d")
    alldaysnum=alldaysnum+(SAMKDAY.days+1)
alldaysnum


# In[9]:


def TimeCal( entrytime, exittime):
    timestr1 = entrytime # '2019-01-03 00:30:00'格式
    timestr2 = exittime
    
    entryTime = datetime.datetime.strptime(timestr1, "%Y-%m-%d %H:%M:%S")
    exitTime = datetime.datetime.strptime(timestr2, "%Y-%m-%d %H:%M:%S")#表示到秒，用于记录时间
    totaltime = exitTime-entryTime
    
    entryday = datetime.datetime.strptime(timestr1.split(' ')[0], "%Y-%m-%d")#表示到天用于计天数
    exitday = datetime.datetime.strptime(timestr2.split(' ')[0], "%Y-%m-%d")
    
    smarkingday = exitday-entryday
    daynums = smarkingday.days+1
    #print(daynums)
    daysarr = []#标定日期
    for i in np.arange(daynums):
       daysarr.append(str(entryday+datetime.timedelta(days=int(i))))#不支持i对应的numpy下的int格式需要用int()转化
    
    caldaysarr = daysarr.copy()#使用副本，表示赋值而不是指针指向同一位置   
    caldaysarr[0] = timestr1
    caldaysarr.append(timestr2)#计算时用到的所有日期信息
    
    daystime=[]#标定日期下对应的占用时间（以秒来划分）
    daysrate = []
    for i in np.arange(daynums):
        starttime = datetime.datetime.strptime(caldaysarr[i], "%Y-%m-%d %H:%M:%S")
        endtime = datetime.datetime.strptime(caldaysarr[i+1], "%Y-%m-%d %H:%M:%S")
        timeinday = endtime-starttime
        daystime.append(timeinday.total_seconds()/3600)
        daysrate.append(timeinday.total_seconds()/totaltime.total_seconds())
    
    return daysarr,daystime,daysrate


# In[10]:


sumdf = pd.DataFrame(columns=['date','rate','Revenue','Gross Revenue','One-Hot Rate','One-Hot Type'])
for i in np.arange(len(smarkingdata)): 
    entt = smarkingdata.loc[i]['ENTRY']
    extt = smarkingdata.loc[i]['EXIT']
    rev = smarkingdata.loc[i]['Revenue']
    gross_rev = smarkingdata.loc[i]['Gross Revenue']
    onehot_rate = smarkingdata.loc[i]['One-Hot Rate']
    onehot_type = smarkingdata.loc[i]['One-Hot Type']
    darr,dtime,drate =  TimeCal(entt,extt)
    for j in np.arange(len(darr)):
        sumdf=sumdf.append({'date':darr[j],'time':dtime[j],'rate':drate[j],'Revenue':rev,'Gross Revenue':gross_rev,
                            'One-Hot Rate':onehot_rate,'One-Hot Type': onehot_type},ignore_index=True)


# In[11]:


#sumdf = pd.DataFrame(columns=['date','rate','Revenue','Gross Revenue'])
sumdf 


# In[12]:


comdf = sumdf.copy()
raterevenue =[]
rategrossrevenue = []
for i in np.arange(len(comdf)):
    raterevenue.append(comdf.loc[i]['rate']*comdf.loc[i]['Revenue'])
    rategrossrevenue.append(comdf.loc[i]['rate']*comdf.loc[i]['Gross Revenue'])


# In[13]:


comdf['RateRevenue'] = raterevenue
comdf['RateGrossRevenue'] = rategrossrevenue


# In[14]:


comdf


# In[15]:


databydaydf = comdf.groupby('date').sum().drop(['rate','Revenue','Gross Revenue'],axis=1)
databydaydf=databydaydf.reset_index()


# In[35]:


onehotRs = []
onehotTs = []
for i in range(len(databydaydf)):  
    datasum = comdf[comdf['date']==databydaydf.loc[i]['date']].sum()
    onehotRs.append(datasum ['One-Hot Rate'])
    onehotTs.append(datasum ['One-Hot Type'])
databydaydf['One-Hot Rate']=onehotRs
databydaydf['One-Hot Type']=onehotTs


# In[17]:


databydaydf


# In[18]:


databydaydf['aveRevenue'] =databydaydf['RateRevenue']/databydaydf['time']
databydaydf['aveGrossRevenue'] =databydaydf['RateGrossRevenue']/databydaydf['time']
databydaydf.to_csv('D:\sjtu\博1\课程\交通运输经济学\daterevenue.csv',index = False)


# In[19]:


databydaydf


# In[20]:


dropdata = databydaydf.drop(databydaydf[databydaydf['aveRevenue']<0.5].index)


# In[21]:


traindata = dropdata.copy()
traindata.pop('date')
traindata.pop('RateRevenue')
traindata.pop('RateGrossRevenue')
traindata


# In[25]:


traindata.to_json(r'D:\sjtu\博1\课程\交通运输经济学\traindata.json')#向量数据导出后csv格式存在问题，采用json或二进制格式保存


# In[ ]:


days =np.arange(len(dropdata))


# In[ ]:


fig,ax = plt.subplots(figsize=(24,8))
ax.plot(days,dropdata['time'])
ax.tick_params(labelsize = 18)
ax.set_xlabel('Time/days',fontsize=20)
ax.set_ylabel('Time/hours',fontsize=20)
ax.set_xlim(left=0)


# In[ ]:


fig,ax = plt.subplots(figsize=(24,8))
ax.plot(days,dropdata['RateRevenue'],label='total revenue during a day')
ax.plot(days,dropdata['RateGrossRevenue'],label='total gross revenue during a day')
ax.legend(fontsize = 18)
ax.tick_params(labelsize = 18)
ax.set_xlabel('Time/days',fontsize=20)
ax.set_ylabel('total profit in a day',fontsize=20)
ax.set_xlim(left=0)


# In[ ]:


fig,ax = plt.subplots(figsize=(24,8))
ax.plot(days,dropdata['RateGrossRevenue']-dropdata['RateRevenue'],label='The difference between revenue and gross revenue')
ax.legend(fontsize = 18)
ax.tick_params(labelsize = 18)
ax.set_xlabel('Time/days',fontsize=20)
ax.set_ylabel('profit in a day',fontsize=20)
ax.set_xlim(left=0)


# In[ ]:


fig,ax = plt.subplots(figsize=(24,8))
ax.plot(days,dropdata['aveRevenue'],label='average revenue during a day')
ax.plot(days,dropdata['aveGrossRevenue'],label='average gross revenue during a day')
ax.legend(fontsize = 18)
ax.tick_params(labelsize = 18)
ax.set_xlabel('Time/days',fontsize=20)
ax.set_ylabel('profit in a day',fontsize=20)
ax.set_xlim(left=0)


# In[ ]:


fig,ax = plt.subplots(figsize=(24,8))
ax.plot(days,dropdata['aveGrossRevenue']-dropdata['aveRevenue'],label='difference between average revenues')
ax.legend(fontsize = 18)
ax.tick_params(labelsize = 18)
ax.set_xlabel('Time/days',fontsize=20)
ax.set_ylabel('profit',fontsize=20)
ax.set_xlim(left=0)


# In[ ]:


fig=plt.figure(figsize=(8,8))
fig.clf()
ax=fig.add_subplot(1,1,1)
ax.hist2d(dropdata['aveRevenue'],dropdata['RateRevenue'],bins=(20,20))
plt.show()


# In[ ]:


fig=plt.figure(figsize=(8,8))
fig.clf()
ax=fig.add_subplot(1,1,1)
ax.scatter(dropdata['aveRevenue'],dropdata['RateRevenue'])
plt.show()


# In[ ]:


fig=plt.figure(figsize=(12,12))
fig.clf()
ax=fig.add_subplot(1,1,1)
ax.scatter(dropdata['aveRevenue'],dropdata['aveGrossRevenue'])
plt.show()


# In[ ]:




