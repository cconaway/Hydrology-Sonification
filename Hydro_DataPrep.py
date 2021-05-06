#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from functools import reduce

import subprocess as sp 
from subprocess import Popen
import os

import sys 
sys.path.append('/home/akh/aesthetic/soni/datamovies_y21/modules')

import writeCmixSco_WT_ac as wRT_wt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#load the hydrologic timeseries data. 
#We have precipication at the HJ Andrews Experimental Forest, and then discharge at two tributaries of the Mckenzie with different percentages of upstream young basalt

#Lookout Creek is "old cascades" endmember, with no young rocks in the catchment 
#Clear Lake is the "young cascades" endmember, largely fed by springs from recent lava flows

dfp0 = pd.read_excel ("Precip_HJASite101.xlsx", parse_dates=[7])
#print (df)
print("we have " + str(dfp0.size) + " precip measurements")

dfc0 = pd.read_excel ("ClearLake.xls",header=27)
#print (df)
print("we have " + str(dfc0.size) + " data entries (uncorrected discharge) at Clear Lake")

#for this one we use the parse_dates function to combine columnes into a datetime column at the outset! 
dfl0 = pd.read_excel ("LOOKOUT_GSLOOK.xlsx", parse_dates= {"date" : ["year","month","day"]})
#dfl0 = pd.read_excel ("Uncorrected_Discharge/HF00402_LOOKOUT.xlsx",sheet_name='HJA', parse_dates= {"date" : ["year","month","day"]})
#print (df)
print("we have " + str(dfl0.size) + " data entries (uncorrected discharge) at Lookout Ck")

LKOUTArea=62.42 #km^2
LAKEArea=239 #km^2

# In[3]:


print(dfp0.head(3))
print(dfc0.head(3))
print(dfl0.head(3))


# In[4]:


#we need to sort the data by data
dfp=dfp0.sort_values(by=['date'])
dfc=dfc0.sort_values(by=['date'])
dfl=dfl0.sort_values(by=['date'])
# rename some of the columns that we will be working with
dfp = dfp.rename(columns={'PRECIP_TOT_DAY': 'HJA Precipitation in mm/day'})
dfc = dfc.rename(columns={'discharge': 'Clear Lake discharge in cfs'})
dfl = dfl.rename(columns={'MEAN_Q_cfs': 'Lookout Ck discharge in cfs'})


#check that the sampling rate is same
dtp = (dfp.date[1] - dfp.date[0])
dtl = (dfl.date[1] - dfl.date[0])
dtc = (dfc.date[1] - dfc.date[0])
print(dtp,dtc,dtl)


# In[33]:


fig,ax = plt.subplots(3,figsize=(15,8));
ax[0]=dfp.plot(x="date", y="HJA Precipitation in mm/day",ax=ax[0]);
ax[1]=dfc.plot(x="date", y="Clear Lake discharge in cfs",ax=ax[1]);
ax[2]=dfl.plot(x="date", y="Lookout Ck discharge in cfs",ax=ax[2]);


#plt.show()
print(dfp['date'].max())
print(dfc['date'].max())
print(dfl['date'].max())


# In[6]:


#find common time frame between the datasets
start = '1958-01-01'
end = '2008-01-01'

conditionP = (dfp['date'] > start) & (dfp['date'] <= end)
dp=dfp.loc[conditionP]
conditionL = (dfl['date'] > start) & (dfl['date'] <= end)
dl=dfl.loc[conditionL]
conditionC = (dfc['date'] > start) & (dfc['date'] <= end)
dc=dfc.loc[conditionC]

print(sum(conditionP==True),sum(conditionC==True),sum(conditionL==True))
print(dp.shape[0],dc.shape[0],dl.shape[0])


# In[7]:


#plot it up
fig,ax = plt.subplots(3,figsize=(15,8));
ax[0]=dp.plot(x="date", y="HJA Precipitation in mm/day",ax=ax[0]);
ax[1]=dc.plot(x="date", y="Clear Lake discharge in cfs",ax=ax[1]);
ax[2]=dl.plot(x="date", y="Lookout Ck discharge in cfs",ax=ax[2]);
fig.savefig('Cascades_Q_Lookout_ClearLake.png')


# In[8]:


# compile the list of dataframes we want to merge
data_frames = [dp, dc, dl]
#merge them using the common date column
df_mergedALL = reduce(lambda  left,right: pd.merge(left,right,on=['date'],
                                            how='outer'), data_frames)

#then reduce clutter by making a final dataframe that just has the quantities of interest
df_pcl = df_mergedALL[['date','HJA Precipitation in mm/day', 'Clear Lake discharge in cfs', 'Lookout Ck discharge in cfs']].copy()
df_pcl.head(3)


# In[9]:


#lets make this a little more readable
#df_norm=(df_pcl-df_pcl.min())/(df_pcl.max()-df_pcl.min())
df_norm=df_pcl.copy()
df_norm["Clear Lake discharge in cfs"] = df_norm["Clear Lake discharge in cfs"]  / df_norm["Clear Lake discharge in cfs"].abs().max()
df_norm["Lookout Ck discharge in cfs"] = df_norm["Lookout Ck discharge in cfs"]  / df_norm["Lookout Ck discharge in cfs"].abs().max() + .35
df_norm["HJA Precipitation in mm/day"] = df_norm["HJA Precipitation in mm/day"]  / df_norm["HJA Precipitation in mm/day"].abs().max() + .65


fig,ax = plt.subplots(figsize=(15,8));
ax=df_norm.plot(x="date", y=["Clear Lake discharge in cfs", "Lookout Ck discharge in cfs", "HJA Precipitation in mm/day"],ax=ax, alpha=0.75);


# In[10]:


#zoom in a bit to see the details
start2 = '1995-08-01'
end2 = '1996-08-01'
#end2 = '1997-08-01'

condition = (df_norm['date'] > start2) & (df_norm['date'] <= end2)
df_norm2=df_norm.loc[condition]
df_2=df_pcl.loc[condition]

fig,ax = plt.subplots(1,2,figsize=(20,8));
ax[1]=df_norm2.plot(x="date", y=["Clear Lake discharge in cfs", "Lookout Ck discharge in cfs", "HJA Precipitation in mm/day"],ax=ax[1], alpha=0.75);
ax[0]=df_2.plot(x="date", y=["Clear Lake discharge in cfs", "Lookout Ck discharge in cfs"],ax=ax[0], alpha=0.75);

ax[0].title.set_text('Un-normalized discharge in CFS')
ax[1].title.set_text('Normalized discharge + HJA precipitation')

fig.savefig('Cascades_Q_Lookout_ClearLake_norm.png')


# In[11]:


## Calculate the dQ
dQLKOT=(np.diff(df_norm2["Lookout Ck discharge in cfs"].values/LKOUTArea, prepend=0))
dQLAKE=(np.diff(df_norm2['Clear Lake discharge in cfs'].values/LAKEArea, prepend=0))
# In[12]:


## Calculate the logical of the signs
thres=0.009
dQbydQ1= abs(dQLKOT-dQLAKE-np.mean(dQLKOT)-np.mean(dQLAKE))>thres
dQbydQ2= ~(np.sign(dQLKOT) == np.sign(dQLAKE))

dQbydQ= (dQbydQ1==True) #& (dQbydQ2==True)
fig,ax=plt.subplots(2)
ax[0].plot(df_norm2['date'], dQLKOT)
ax[0].plot(df_norm2['date'], dQLAKE)

ax[1].plot(dQbydQ)

print(np.sum(dQbydQ))


# In[13]:


data=np.log10(df_norm2['Lookout Ck discharge in cfs']/LKOUTArea)

# make an array of integers: 
v=3
k = np.arange(12+1)
print(k)
root=220
print('major scale:')
# intervals 2,2,1,2,2,2,1 
major = np.array([0,2,4,5,7,9,11,12])
print(major)
# Octaves 1,v=0
v=0

f=np.array(root*2**(v+major/12))
for v in [1,2]: # Octaves 2,3,4
    f = np.append(f,root*2**(v+major/12))
print(f)
npitches=len(f)
print(len(f))


# In[14]:


#now bin and interp
print("Data length", len(data))
notes=len(data)
dur_sound=1/16
dur_score = notes*dur_sound
print("Duration score", dur_score)

tscore=np.linspace(0,dur_score,notes)


# In[15]:


## Bin data 
bins=np.linspace(min(data),max(data), npitches)

p=np.digitize(data,bins)

datapitches=[ f[i-1] for i in p ]

plt.hist(datapitches)


# In[16]:


# (4) write RTcmix Wavetable score

base_name = 'test_CFS'

tones_dict = {}
tones_dict['times'] = np.asarray(tscore)
tones_dict['notes'] = np.asarray(datapitches)
tones_dict['durs'] = np.ones(len(datapitches))*dur_sound # the 0.8 makes for more discrete pitches
tones_dict['amps'] = np.ones(len(datapitches))*2000
tones_dict['pans'] = np.ones(len(datapitches))*0.5

score_name = wRT_wt.writesco(tones_dict,base_name)


# In[17]:


## Now try speeding it up based on dQ 

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

#scale DQ to mean and std
dQ2=(dQLKOT-np.mean(dQLKOT))/np.std(dQLKOT)
threshold=2

#Find moving average to speed it up
dQabs=abs(dQ2)

## smooth it out 



fig,ax=plt.subplots()
scale=1/5

dt=dur_sound*(1+dQabs**(scale)*4)

dt=moving_average(dt, 7) #smooth out
dt=np.array(dt)
dt[np.isnan(dt)]=dur_sound # replace nan
tscore2=np.cumsum(dt)
dt[dt<dur_sound]=dur_sound # get rid of edge effects

ax.plot(tscore2,dt)
ax.hlines(dur_sound, 0, max(tscore2) ,'red')



# In[18]:


# (4) write RTcmix Wavetable score

base_name = 'test_CFS_speedup'

print(len(tscore), len(datapitches), len(dt))
tones_dict = {}
tones_dict['times'] = np.asarray(tscore2)
tones_dict['notes'] = np.asarray(datapitches)
tones_dict['durs'] = np.asarray(dt) # the 0.8 makes for more discrete pitches
tones_dict['amps'] = np.ones(len(datapitches))*2000
tones_dict['pans'] = np.ones(len(datapitches))*0.5

score_name = wRT_wt.writesco(tones_dict,base_name)


# In[19]:


## speedupDQ 
def SpeedUpDQ(dQ, scale, dur_sound):

    #scale DQ to mean and std
    dQ2=(dQ-np.mean(dQ))/np.std(dQ)
    threshold=2

    #Find moving average to speed it up
    dQabs=abs(dQ2)
    dt=dur_sound*(1+dQabs**(scale)*4)

    dt=moving_average(dt, 7) #smooth out
    dt=np.array(dt)
    dt[np.isnan(dt)]=dur_sound # replace nan
    tscore=np.cumsum(dt)
    dt[dt<dur_sound]=dur_sound # get rid of edge effects

    return tscore, dt

def freqDigitize(data, root, noctaves,v=0, mode='ionian'):
    
    modes={
        'ionian':[2,2,1,2,2,2,1],
        'dorian':[2,1,2,2,2,1,2],
        'phyrigian':[1,2,2,2,1,2,2],
        'lydian':[2,2,2,1,2,2,1],
        'mix':[2,2,1,2,2,1,2],
        'aeol':[2,1,2,2,1,2,2]
    }
    
    
    data=np.log(data)
    major=np.cumsum(modes[mode])
    f=np.array(root*2**(v+major/12))
    major1=major
    v1=np.zeros(len(major))
    
    for vi in range(1,noctaves):
        print(vi)
        for mi in range(len(major)):
            f = np.append(f, root*2**((vi+v)+major[mi]/12))
            v1=np.append(v1,v)
        major1=np.append(major1, major)
    
    
    npitches=len(f)
    bins=np.linspace(min(data), max(data), npitches)
    p=np.digitize(data, bins)
    datapitches=[f[i-1] for i in p ]
    majortone=[major1[i-1] for i in p]
    octive=[v1[i-1] for i in p]
    
    return datapitches, majortone, octive


# In[20]:


def semiToneAddOn(databool, k1, v1, root):

    knew1=k1+0.5
    knew1[knew1>12]=0
    v1[knew1>12]=v1[knew1>12]+1
    s1=root*2**(v1+knew1/12)
    s1=s1[databool]
    return s1

def CalcSemiTone(nTones, k1, v1, root):
    
    n=np.arange( min(0,nTones), max(0,nTones),1)
    knew1=k1+n
    knew1[knew1>12]=0
    knew1[knew1<0]=12
    v1=np.ones(len(n))*v1
    v1[knew1>12]=v1[knew1>12]+1
    v1[knew1<0]=v1[knew1<0]-1
    
    s1=root*2**(v1+knew1/12)

    return s1

def MultiSemiToneAddOn(notelength, thres, dQ1,dQ2, k1, k2, v1, v2, root1, root2):
    dQbydQ=(dQ1-np.mean(dQ1))-(dQ2-np.mean(dQ2))
    bins=np.linspace(min(dQbydQ), max(dQbydQ), 6)
    ntones=np.digitize(dQbydQ,bins)*np.sign(dQbydQ)
    pitches=[]
    totaltime=[]
    time=np.linspace(0,len(dQ1)*notelength, len(dQ1))
    for i in range(len(dQ1)):
        if abs(dQbydQ[i]) > thres: 
            d1= CalcSemiTone(ntones[i], k1[i], v1[i], root1)
            d2= CalcSemiTone(ntones[i], k2[i], v2[i], root2)
            t1= np.ones(int(2*abs(ntones[i])))*time[i]
            print(int(2*abs(ntones[i])), len(d1), ntones[i])
            pitches=np.concatenate((pitches,d1))
            pitches=np.concatenate((pitches,d2))
            totaltime=np.concatenate((totaltime, t1))
            
            

    return pitches, totaltime

# In[21]:


## Speed up based on average of LKOT and LAKE 

dQAvg=(dQLKOT+dQLAKE)/2
scale =1/5
tscore, dt=SpeedUpDQ(dQAvg, scale, dur_sound)


# In[22]:


# (4) write RTcmix Wavetable score

base_name = 'test_CFS_speedup'

print(len(tscore), len(datapitches), len(dt))
tones_dict = {}
tones_dict['times'] = np.asarray(tscore)
tones_dict['notes'] = np.asarray(datapitches)
tones_dict['durs'] = np.asarray(dt) # the 0.8 makes for more discrete pitches
tones_dict['amps'] = np.ones(len(datapitches))*2000
tones_dict['pans'] = np.ones(len(datapitches))*0.5

score_name = wRT_wt.writesco(tones_dict,base_name)


# In[31]:


rootLKOUT=349.23
rootLAKE=261.63
notelength=1/4

pitchesLKOUT, k1,v1=freqDigitize(df_norm2['Lookout Ck discharge in cfs']/LKOUTArea, 349.23, 3, mode='lydian')
pitchesLAKE, k2,v2=freqDigitize(df_norm2['Clear Lake discharge in cfs']/LAKEArea, 261.63, 3)
df=pd.DataFrame({"pitches":pitchesLKOUT})
df.to_csv("LOOKOUT_PITCHES.csv")

pitchesStream=np.append(pitchesLKOUT, pitchesLAKE)
pitches=np.append(pitchesLKOUT, pitchesLAKE)

tscore1=np.linspace(0, len(pitches)*(notelength)/2, int(len(pitches)/2))
tscore=np.tile(tscore1,2)


# In[28]:


# (4) write RTcmix Wavetable score

base_name = 'test_CFS_both'


print(len(tscore), len(pitches))
tones_dict = {}
tones_dict['times'] = np.asarray(tscore)
tones_dict['notes'] = np.asarray(pitches)
tones_dict['durs'] = np.ones(len(pitches))*(notelength) # the 0.8 makes for more discrete pitches
tones_dict['amps'] = np.ones(len(pitches))*2000
tones_dict['pans'] = np.ones(len(pitches))*0.5

score_name = wRT_wt.writesco(tones_dict,base_name)


# In[29]:


s1=semiToneAddOn(dQbydQ, np.array(k1), np.array(v1), rootLKOUT)
s2=semiToneAddOn(dQbydQ,  np.array(k2), np.array(v2), rootLAKE)

fig,ax=plt.subplots(3)
ax[0].plot(pitchesLKOUT)
ax[1].plot(tscore1[dQbydQ==1],s1, '.')
ax[1].plot(tscore1[dQbydQ==1], s2, '.')
ax[1].set_xlim([0,len(pitches)*notelength*.5])
ax[2].set_xlim([0, len(pitches)*notelength*.5])
ax[2].scatter(tscore1, dQbydQ, c=dQbydQ)
print(len(pitchesLKOUT))
plt.show()

tscore_ALL1=np.append(tscore, tscore1[dQbydQ])
tscore_ALL=np.append(tscore_ALL1,tscore1[dQbydQ])
pitches1=np.append(pitches, s1)
pitches3=np.append(pitches1,s2)
print(len(s1))
print(len(s2))

#print(max(pitchesLKOUT[dQbydQ]))


# In[30]:


# (4) write RTcmix Wavetable score

base_name = 'test_CFS_diso1'

print(len(tscore_ALL), len(pitches3))
tones_dict = {}
tones_dict['times'] = np.asarray(tscore_ALL)
tones_dict['notes'] = np.asarray(pitches3)
tones_dict['durs'] = np.ones(len(pitches3))*notelength*.8 # the 0.8 makes for more discrete pitches
tones_dict['amps'] = np.ones(len(pitches3))*2000
tones_dict['pans'] = np.ones(len(pitches3))*0.5

score_name = wRT_wt.writesco(tones_dict,base_name)


# In[ ]:
thres=0.0001
pAddOn,tAddOn=MultiSemiToneAddOn(.25, thres, dQLKOT, dQLAKE, k1, k2, v1, v2, rootLKOUT, rootLAKE)
plt.scatter(tAddOn,pAddOn)

tscore1=np.linspace(0, len(pitchesStream)*(notelength)/2, int(len(pitchesStream)/2))
tscore=np.tile(tscore1,2)

tscore_ALL=np.append(tscore,tAddOn)
pitches=np.append(pitchesStream,pAddOn)
durs=np.ones(len(pitches))*notelength
# In[30]:


# (4) write RTcmix Wavetable score

base_name = 'test_CFS_disoAddOn'

print(len(tscore_ALL), len(pitches))
tones_dict = {}
tones_dict['times'] = np.asarray(tscore_ALL)
tones_dict['notes'] = np.asarray(pitches)
tones_dict['durs'] = np.ones(len(pitches))*notelength*.8 # the 0.8 makes for more discrete pitches
tones_dict['amps'] = np.ones(len(pitches))*2000
tones_dict['pans'] = np.ones(len(pitches))*0.5

score_name = wRT_wt.writesco(tones_dict,base_name)


