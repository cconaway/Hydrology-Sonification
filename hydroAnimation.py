#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
from functools import reduce

import subprocess as sp 
from subprocess import Popen
import os

import matplotlib.animation as animation
import matplotlib.patches as patches
from IPython.display import HTML

import sys 
sys.path.append('/home/akh/aesthetic/soni/datamovies_y21/modules')

import writeCmixSco_WT_ac as wRT_wt


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

print(dfl0.columns)

print(dfc0.columns)
print(dfp0.columns)

LKOUTArea=62.42 #km^2
LAKEArea=239 #km^2
notelength=1/4

# In[3]:


#find common time frame between the datasets
start = '1995-08-01'
end = '1996-08-01'

conditionP = (dfp0['date'] > start) & (dfp0['date'] <= end)
dp=dfp0.loc[conditionP]
conditionL = (dfl0['date'] > start) & (dfl0['date'] <= end)
dl=dfl0.loc[conditionL]
conditionC = (dfc0['date'] > start) & (dfc0['date'] <= end)
dc=dfc0.loc[conditionC]

print(sum(conditionP==True),sum(conditionC==True),sum(conditionL==True))
print(dp.shape[0],dc.shape[0],dl.shape[0])


# In[8]:


fig,ax=plt.subplots()
dQl=abs(np.diff(dl['MEAN_Q_cfs'].values/LKOUTArea, prepend=0))
dQc=abs(np.diff(dc['discharge'].values/LAKEArea, prepend=0))
#dP=abs(np.diff(dp['PRECIP_TOT_DAY']), prepend=0)
print(len(dQl))
print(len(dc))

ax.scatter(np.log10(dl['MEAN_Q_cfs']/LKOUTArea),np.log10(dQl), c='red', edgecolor='black', s=dp['PRECIP_TOT_DAY'])
ax.scatter(np.log10(dc['discharge']/LAKEArea), np.log10(dQc), c='blue', edgecolor='black', s=dp['PRECIP_TOT_DAY'])
#ax.scatter(np.log())
ax.set_xlabel('ln(Q)')
ax.set_ylabel('ln(dQ)')


# In[5]:


""" Update the graph using color map with 0 as white and 1 as red
Then iterate over the color values for each timesep by multiplying by .95
"""


from matplotlib.colors import LinearSegmentedColormap

fig,ax=plt.subplots()

def dQFigureSetup(ax):
    ax.set_xlabel('ln(Q)')
    ax.set_ylabel('ln(dQ)')
    ax.set_yscale('log')
    ax.set_xscale('log')
            
    ax.set_ylim([0.1, 10**8.5])
    ax.set_xlim([0.1,10**9])
    
    return ax

Qc=np.array(dc['discharge'].values)
Ql=np.array(dl['MEAN_Q_cfs'].values)
clist=[]
llist=[]
colors=[[0,0,1,0], [0,0,1,0.5], [0,0.2,0.4,1]]
cmap=LinearSegmentedColormap.from_list("", colors)
x_vals=[]
y_vals=[]
intensity=[] #np.array([1])

scatter=ax.scatter(x_vals, y_vals, c=[], cmap=cmap, vmin=0, vmax=1) #, s=dp['PRECIP_TOT_DAY'])

t=np.arange(1,365)#len(dQc)-2)

print(len(t), "days animating")
print("notes/frames per second", 2)
interval=notelength*1000
i=0
def updateOneDot(t):
    global x_vals, y_vals, i, intensity, scatter
    x_vals.extend([Qc[i]])
    y_vals.extend([dQc[i]])
    
    scatter.set_offsets(np.c_[x_vals, y_vals])
    
    intensity=np.concatenate((np.array(intensity)*.96, np.array([1])))

    scatter.set_array(np.array(intensity))

    ax.set_title('Time: '%t)
    #print(i, intensity, x_vals,y_vals)
    i=i+1
    
#ani=animation.FuncAnimation(fig, updateOneDot, frames=t, interval=interval)

#HTML(ani.to_html5_video())
#ani.save("hydro04252021.mp4")

#for i in range(len(dQc)):
    
#    dQQc=[Qc[i], dQc[i]]
#    cdot=patches.Circle(dQQc,fc="orange", ec="black")
#    clist.append(cdot)
#    dQQl=[Ql[i], dQl[i]]
#    ldot=patches.Circle(dQQl, fc='blue', ec="black")
#    llist.append(ldot)


# In[187]:

notelength=1/4 #s
xl=(np.log(min(dc['discharge'])))
xl2=(np.log(min(dl['MEAN_Q_cfs'])))

xu=(np.log(max(dc['discharge'])))
xu2=(np.log(max(dl['MEAN_Q_cfs'])))

xu=max(xu,xu2)


print(max(np.log(dQc)))
print(max(np.log(dQl)))


# In[210]:
""" To do two dots use a colormap with -1 and 1 as the range respectively red and blue 
with zero as white then multiply by .96 each time so they get closer to 0

Set dot size using scatter.set_sizes to  
1) set dot size proportional to precipitation 
2) set dot size for each time step to preciptation so they pulse with precip
    
"""
fig,ax=plt.subplots()

ax.set_xlabel('ln(Q)')
ax.set_ylabel('ln(dQ)')
ax.set_yscale('log')
ax.set_xscale('log')
        

Qc=np.array(dc['discharge'].values+.001)/LAKEArea
Ql=np.array(dl['MEAN_Q_cfs'].values+.001)/LKOUTArea

ax.set_ylim([.001, 100])
ax.set_xlim([.01,10**2.5])

Pre=np.array(dp['PRECIP_TOT_DAY'].values)
clist=[]
llist=[]
colors=[ [.8,.2,0,1], [.6,.1,0,.5], [0,0,0,0], [0,.1,0.6,0.5], [0,.2,0.8,1]]
cmap=LinearSegmentedColormap.from_list("", colors)
x_vals=[]
y_vals=[]
intensity=[] #np.array([1])
size=[]

scatter=ax.scatter(x_vals, y_vals, c=[], cmap=cmap, vmin=-1, vmax=1) #, s=dp['PRECIP_TOT_DAY'])

t=np.arange(1,365)#len(dQc)-2)

print(len(t), "days animating")
print("notes/frames per second", 1/notelength)
interval=notelength*1000
i=0

def updateTwoDots(t):
    global x_vals, y_vals, i, intensity, scatter
    x_vals.extend([Ql[i], Qc[i]])
    y_vals.extend([dQl[i], dQc[i]])
    size.extend([Pre[i]+20,Pre[i]+20])
    #size=np.ones(len(x_vals))*(Pre[i]+15)
    scatter.set_offsets(np.c_[x_vals, y_vals])
    scatter.set_sizes(size)
    
    intensity=np.concatenate((np.array(intensity)*.95, np.array([1,-1])))

    scatter.set_array(np.array(intensity))

    ax.set_title('Time: %0.0f' %t)
    #print(i, intensity, x_vals,y_vals)
    i=i+1

def updateTwoDotsPulse(t):
    global x_vals, y_vals, i, intensity, scatter
    x_vals.extend([Ql[i], Qc[i]])
    y_vals.extend([dQl[i], dQc[i]])
    size=np.ones(len(x_vals))*(Pre[i]+20)
    scatter.set_offsets(np.c_[x_vals, y_vals])
    scatter.set_sizes(size)
    
    intensity=np.concatenate((np.array(intensity)*.95, np.array([1,-1])))

    scatter.set_array(np.array(intensity))

    ax.set_title('Time: %0.0f' %t)
    #print(i, intensity, x_vals,y_vals)
    i=i+1
       
    
#ani=animation.FuncAnimation(fig, updateTwoDotsPulse, frames=t, interval=interval)

#HTML(ani.to_html5_video())
#ani.save("hydro04272021_bOTH.mp4")

# In[ ]:
""" Now combine them and update multiple plots in the same figure"""

figM=plt.figure(constrained_layout=True)
widths=[2,3]
heights=[1,5]
spec=figM.add_gridspec(2,2)#(ncols=2, nrows=2, width_ratios=widths,
                          # height_ratios=heights)

ax1=figM.add_subplot(spec[0,0])
ax2=figM.add_subplot(spec[1,0])
ax3=figM.add_subplot(spec[0:,-1])

ax1.plot(timeQ,np.log10(Ql), c=[.6,.1,0,.5])
ax1.plot(timeQ,np.log10(Qc), c=[0,.1,0.6,0.5])
ax1.set_ylabel("Discharge")

timeQ=np.arange(0,len(Qc))
ax2.plot(timeQ, Pre, c="grey")
#ax2.invert_xaxis()
#ax2.invert_yaxis()
ax2.set_xlabel("Precip")
#ax2.xaxis.set_ticks_position('top')
#ax2.xaxis.set_label_position('top')

ax3.set_xlabel('ln(Q)')
ax3.set_ylabel('ln(dQ)')
ax3.set_yscale('log')
ax3.set_xscale('log')
        

Qc=np.array(dc['discharge'].values+.001)/LAKEArea
Ql=np.array(dl['MEAN_Q_cfs'].values+.001)/LKOUTArea

ax3.set_ylim([.001, 100])
ax3.set_xlim([.01,10**2.5])


line=[]
x=[]
yl=[]
yc=[]
yp=[]
colors=[ [.8,.2,0,1], [.6,.1,0,.5], [0,0,0,0], [0,.1,0.6,0.5], [0,.2,0.8,1]]
cmap=LinearSegmentedColormap.from_list("", colors)
x_vals=[]
y_vals=[]
intensity=[] #np.array([1])
size=[]
ecalpha=[]

scatter=ax3.scatter(x_vals, y_vals, c=[], cmap=cmap, vmin=-1, vmax=1) #, s=dp['PRECIP_TOT_DAY'])


i=0

def lineColorChange(t):
    global x, yc, yl, i, line1,line2, line3, yp, x_vals, y_vals, intensity, size, ecalpha
    x.append(timeQ[i])
    yl.append(np.log10(Ql[i]+0.00001))
    yc.append(np.log10(Qc[i]+0.00001))
    yp.append(Pre[i])

    line, = ax1.plot(x,yl, color=[.8,.2,0,1])
    line2, = ax1.plot(x,yc, color=[0,.2,0.8,1])
    line3=ax2.plot(x, yp, color='green')
    ecalpha=np.concatenate( (np.array(ecalpha)*.95, np.array([1])))
    ec=np.zeros((len(ecalpha),4))
    ec[:,3]=np.array(ecalpha)*.92
    
    #line, = ax1.plot(timeQ[i], np.log10(Ql[i]), c="red")
    

    x_vals.extend([Ql[i], Qc[i]])
    y_vals.extend([dQl[i], dQc[i]])
    size=np.ones(len(x_vals))*(Pre[i]+20)
    scatter.set_offsets(np.c_[x_vals, y_vals])
    scatter.set_sizes(size)
    scatter.set_edgecolors(ec)
    
    intensity=np.concatenate((np.array(intensity)*.95, np.array([1,-1])))

    scatter.set_array(np.array(intensity))

    ax3.set_title('Time: %0.0f' %t)
    
    i=i+1


ani=animation.FuncAnimation(figM,lineColorChange, frames=t, interval=interval)
ani.save("test.mp4")

#def updateSubplots(): 
    

# In[ ]:




