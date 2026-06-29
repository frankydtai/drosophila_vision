# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:31:47 2024

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 

plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'
    
fs_legend = 8
fs_tick   = 7
fs_axis   = 8
fs_title  = 12

mycolor = 'red'

cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])

data   = np.load('Models/data.npy')
model1 = np.load('Models/best_model_with_Ih.npy')
model2 = np.load('Models/best_model_Ih_turned_off.npy')
model3 = np.load('Models/best_model_no_Ih.npy')

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def Gauss1D(FWHM,RFsize):
    
    myrange=RFsize/2
    sigma=FWHM/(2.0*np.sqrt(2*np.log(2)))
    x=np.arange(-myrange,(myrange+1),1)*1.0
    z=np.exp(-x**2/(2*(sigma**2)))
    z=z/np.sum(z)
    
    return z
    
def Gauss2D(FWHM,RFsize):
    
    myrange=RFsize/2
    sigma=FWHM/(2.0*np.sqrt(2*np.log(2)))
    x=np.arange(-myrange,(myrange+1),1)
    y=np.arange(-myrange,(myrange+1),1)
    x,y=np.meshgrid(x,y)
    r=np.sqrt(x**2+y**2)
    z=np.exp(-r**2/(2*(sigma**2)))
    z=z/np.sum(z)
    
    return z

def rebin(x,f0,f1=0):
    
    mydim=x.ndim
    n=x.shape
    
    if mydim==1:
        
        result=np.zeros((f0))
        if f0 <=  n[0]:
            result=x[0:n[0]:int(n[0]/f0)]
        if f0 >  n[0]:
            result=np.repeat(x,int(f0/n[0]))
            
    if mydim==2:
        
        result=np.zeros((f0,f1))
        interim=np.zeros((f0,n[1]))
        
        #handling 1st dim
        
        if f0 <=  n[0]:
            interim=x[0:n[0]:int(n[0]/f0),:]
        if f0 >  n[0]:
            interim=np.repeat(x,int(f0/n[0]),axis=0)
            
        #handling 2nd dim
        
        if f1 <=  n[1]:
            result=interim[:,0:n[1]:int(n[1]/f1)]
        if f1 >  n[1]:
            result=np.repeat(interim,int(f1/n[1]),axis=1)

    return result.copy()

def blurr(inp_image,FWHM):
    
    if inp_image.ndim==1: z=Gauss1D(FWHM,4*FWHM)
    if inp_image.ndim==2: z=Gauss2D(FWHM,4*FWHM)
    
    result=scipy.ndimage.convolve(inp_image,z)
    
    return result

def plot_model(data, model, label1 = 'data', label2 = 'model'):
    
    mylw = 2
    
    # set x and y position for each cell type
    
    xpos = np.zeros(13)
    ypos = np.zeros(13)
    
    xstep = 0.16
    
    # L1-5
    
    xpos[0:5] = 0.15+np.arange(5)*xstep
    ypos[0:5] = 0.80
    
    # T4-Inputs
    
    xpos[5:9] = 0.15+np.arange(4)*xstep
    ypos[5:9] = 0.5
    
    # T5-Inputs
    
    xpos[9:13] = 0.15+np.arange(4)*xstep
    ypos[9:13] = 0.2
    
    xsize = 0.12
    ysize = 0.08
            
    def set_yticks():
        
        if i == 0 or i == 5 or i == 9:
            plt.yticks(np.arange(5)*15-30,np.arange(5)*15-30,fontsize=fs_tick)
        else:
            plt.yticks(np.arange(5)*15-30,'')
            
    ylabelset = set([0,5,9])
            
    plt.figure(figsize=(7.5,10))
    
    for i in range(13):
                
        # Extract Impulse Responses from xt

        ImpR_model = 1.0*model[i,4]
        
        ImpR_data  = 1.0*data[i,4]
        
        # Extract Receptive field from xt
        
        maxamp_model = np.max(abs(ImpR_model))
        maxamp_data  = np.max(abs(ImpR_data))
        
        maxt_model = np.where(abs(ImpR_model) == np.max(abs(ImpR_model)))[0][0]
        maxt_data  = np.where(abs(ImpR_data)  == np.max(abs(ImpR_data)))[0][0]
        
        RecF_model = rebin(model[i,:,maxt_model],45)
        RecF_model = blurr(RecF_model,5)
        RecF_model = RecF_model/np.max(abs(RecF_model))*maxamp_model
        
        RecF_data  = rebin(data[i,:,maxt_data],45)
        RecF_data  = blurr(RecF_data,5)
        RecF_data  = RecF_data/np.max(abs(RecF_data))*maxamp_data
        
        # --------plotting ---------------------------------------
        
        setmyaxes(xpos[i],ypos[i],xsize,ysize)
        
        plt.plot(np.roll(RecF_data,-2),color='gray',label=label1,linewidth = mylw)
        plt.plot(np.roll(RecF_model,-2),color=mycolor,label=label2,linewidth = mylw)
            
        plt.ylim(-30,30)
        set_yticks()
        
        plt.xlim(0,40)
 
        plt.xticks(np.arange(3)*20,np.arange(3)*20-20,fontsize=fs_tick)
        
        if i == 8: plt.legend(bbox_to_anchor=(1.4, 1),frameon=False,fontsize=fs_legend)
            
        plt.title(cell_list[i],fontsize=fs_title,fontweight='bold')
            
        plt.xlabel('azimuth [$^\circ$]',fontsize=fs_axis)
        
        if i in ylabelset: plt.ylabel('response [mV]',fontsize=fs_axis)
                
        setmyaxes(xpos[i],ypos[i]-0.13,xsize,ysize)
        
        plt.plot(ImpR_data,color='gray',label=label1,linewidth = mylw)
        plt.plot(ImpR_model,color=mycolor,label=label2,linewidth = mylw)
            
        plt.ylim(-30,30)
        set_yticks()
        
        plt.xlim(0,200)
        
        plt.xticks(np.arange(3)*100,np.arange(3)*1.0,fontsize=fs_tick)
            
        plt.xlabel('time [s]',fontsize=fs_axis)
        
        if i in ylabelset: 
            
            plt.ylabel('response [mV]',fontsize=fs_axis)
        
        plt.pause(0.1)
        
plot_model(data, model1)