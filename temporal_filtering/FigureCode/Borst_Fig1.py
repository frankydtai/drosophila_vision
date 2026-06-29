# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:16:48 2016

@author: aborst
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

fs_title = 12
fs_axis  = 8

plt.figure(figsize=(7.5,10))

xsize=0.08
ysize=0.058

xstep=0.02
ystep=0.04
    
size=40
mylw=2

myLAtitle=['L1','L2','L3','L4','L5']
myT4title=['Mi1','Tm3','Mi4','Mi9']
myT5title=['Tm1','Tm2','Tm4','Tm9']

spatialrf=np.zeros((5,size,size))
cellresponse=np.zeros((5,100))
contrastdata=np.zeros((5,7,2))
  
stim=np.zeros(100)
stim[20:99]=1.0

def lowpass(x,tau):
    
    # swaps input dimension such as last dim becomes first
    
    x=x.transpose(np.roll(np.arange(x.ndim),1))  
    n=x.shape[0]
    result=np.zeros_like(x)
    
    if tau<1:
        result=x
    if tau>=1:
        result[0]=x[0]
        for i in range(0,n-1):
            result[i+1]=1.0/tau*(x[i]-result[i])+result[i]
            
    # swaps output dimension such as first dimension becomes last again
            
    result=result.transpose(np.roll(np.arange(result.ndim),-1))
                
    return result
    
def highpass(x,tau): 
    
    result=x-lowpass(x,tau) 
    return result

def normalize(x):
    
    mymax=np.nanmax(x)
    mymin=np.nanmin(x)
    
    if np.abs(mymax)>np.abs(mymin):
        absmax=np.abs(mymax)
    else:
        absmax=np.abs(mymin)
        
    result=x/absmax
    
    if mymax==mymin:
        result=x*0.0
        
    return result

def rect(x,thrld):
    
    result=x-thrld
    result=result*(result>0)
    result=result+thrld
    
    return result

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

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def calc_rf(FWHM_center,FWHM_surrnd,surrnd_fac,polarity):

    center=Gauss2D(FWHM_center,200)
    center=center/np.max(center)
    surrnd=Gauss2D(FWHM_surrnd,200)
    surrnd=surrnd/np.max(surrnd)
    rf=polarity*(center-surrnd_fac*surrnd)
    rf=rf[0:200,0:200]
    rf=rf[int(100-size/2):int(100+size/2),int(100-size/2):int(100+size/2)]
    
    return rf
    
def calc_sr(lp_tau,hp_tau,polarity):
    
    step_response=lowpass(stim,lp_tau)
    if hp_tau !=  0:
        step_response=highpass(step_response,hp_tau)
    step_response=normalize(step_response)
    step_response=polarity*(step_response-0.5)+0.5
    
    return step_response

def calc_data_LA():
    
    global spatialrf,cellresponse
    
    # spatial receptive field parameters

    FWHM_center=[6.31,6.74,5.90,8.55,6.79]
    FWHM_surrnd=[41.1,29.3,15.1,32.7,31.4]
    surrnd_fac=[0.012,0.013,0.19,0.046,0.035]
    polarity=[-1,-1,-1,-1,1]
    
    # time constants * 10 msec
    
    hp_tau=[39.1,28.8,00.0,38.1,12.7]
    lp_tau=[03.8,05.8,05.4,02.3,04.2]
    
    # -------------------------------------
    
    for i in range(5):
        
        spatialrf[i]=calc_rf(FWHM_center[i],FWHM_surrnd[i],surrnd_fac[i],polarity[i])
        cellresponse[i]=calc_sr(lp_tau[i],hp_tau[i],polarity[i])

def calc_data_T4():
    
    global spatialrf,cellresponse
    
    # spatial receptive field parameters

    FWHM_center=[6,12,6,6]
    FWHM_surrnd=[29,7,16,24]
    surrnd_fac=[0.022,0.000,0.132,0.063]
    polarity=[1,1,1,-1]
    
    # time constants * 10 msec
    
    hp_tau=[31.8,26.0,0.00,0.00]
    lp_tau=[05.4,02.7,03.8,07.7]
    
    for i in range(4):
        
        spatialrf[i]=calc_rf(FWHM_center[i],FWHM_surrnd[i],surrnd_fac[i],polarity[i])
        cellresponse[i]=calc_sr(lp_tau[i],hp_tau[i],polarity[i])
        
def calc_data_T5():
    
    global spatialrf,cellresponse
    
    # spatial receptive field parameters

    FWHM_center=[8,8,11,7]
    FWHM_surrnd=[27,31,35,24]
    surrnd_fac=[0.040,0.035,0.054,0.046]
    polarity=[-1,-1,-1,-1]
    
    # time constants * 10 msec
    
    hp_tau=[29.6,15.3,24.9,0.00]
    lp_tau=[04.4,01.4,02.4,10.7]
    
    # -------------------------------
    
    for i in range(4):
        spatialrf[i]=calc_rf(FWHM_center[i],FWHM_surrnd[i],surrnd_fac[i],polarity[i])
        cellresponse[i]=calc_sr(lp_tau[i],hp_tau[i],polarity[i])
        
def set_tick_params():
    
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top=False,         # ticks along the top edge are off
    labeltop=False)    # labels along the bottom edge are off
    
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,       # ticks along the top edge are off
    labelright=False)  # labels along the bottom edge are off
        
def plot_1column(data,response,mytitle,mycolor,xindex,yindex,start):
    
    myt=np.linspace(0,1,100)
    
    mymax=np.max(np.abs(data))
    
    # top row ----------------------------------------------------------
    
    setmyaxes(xindex*(xsize+xstep),1-(yindex+1)*(ysize+ystep),xsize,ysize)
    
    # enhance surround --------------------------------
      
    posdata=rect(data,0)
    negdata=rect(-data,0)
            
    if abs(np.min(data))>abs(np.max(data)):
        enh_data=3*posdata-negdata
    else:
        enh_data=posdata-3*negdata
        
    # -----------------------
    
    plt.imshow(enh_data,cmap='PiYG',vmin=-mymax,vmax=mymax, origin='lower')
    plt.xticks(np.arange(3)*20,np.arange(3)*20-20)
    
    plt.xlabel('azimuth [$^\circ$]',fontsize=fs_axis)
    
    if start==0: 
        
        plt.yticks(np.arange(3)*20,np.arange(3)*20-20)
        plt.ylabel('elevation [$^\circ$]',fontsize=fs_axis)
        
    else:
        
        plt.yticks(np.arange(3)*20,'')

    plt.xlim(0,40)
    plt.ylim(0,40)
    
    set_tick_params()
    
    plt.title(mytitle,color=mycolor,fontsize = fs_title,fontweight='bold')
    
    # second row ------------------------------------------------------------
    
    setmyaxes(xindex*(xsize+xstep),1-(yindex+2)*(ysize+ystep),xsize,ysize)
    
    if np.min(data)<-0.5: data = data + 1.0
    
    plt.plot(data[20], color=mycolor,linewidth=mylw)
    plt.xlabel('azimuth [$^\circ$]',fontsize=fs_axis)
    
    if start==0: 
        
        plt.yticks(np.arange(3)/2.0,np.arange(3)/2.0)
        plt.ylabel('receptive field',fontsize=fs_axis)
        
    else:
        
        plt.yticks(np.arange(3)/2.0,'')
    
    plt.ylim(-0.2,1.2)
    plt.xticks(np.arange(3)*20,np.arange(3)*20-20)
    
    plt.xlim(0,40)
    
    # third row ------------------------------------------------------------
    
    setmyaxes(xindex*(xsize+xstep),1-(yindex+3)*(ysize+ystep),xsize,ysize)
    
    plt.plot(myt,response, color=mycolor,linewidth=mylw)
    plt.xlabel('time [s]',fontsize=fs_axis)
    
    if start==0: 
        
        plt.ylabel('step response',fontsize=fs_axis)
        plt.yticks(np.arange(3)/2.0,np.arange(3)/2.0)
        
    else:
        
        plt.yticks(np.arange(3)/2.0,'')
        
    plt.ylim(-0.2,1.2)
    plt.xlim(0,1)
    
    set_tick_params()
    
def plot_data(nofcells,mytitle,mycolor,xindex,yindex):
    
    global spatialrf,cellresponse
    
    for i in range(nofcells):
        
        data=spatialrf[i]
        response=cellresponse[i]
        
        plot_1column(data,response,mytitle[i],mycolor,xindex+i,yindex,i)
             
        plt.pause(0.01)
        
def plot_cbar():
    
    mybar = np.zeros((100,15))

    for i in range(100):
        mybar[i]=i
        
    setmyaxes(0.89,0.803,0.03,0.06)
    
    plt.imshow(mybar,cmap='PiYG',origin='lower')
    
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    plt.tick_params(
    axis='y',          
    which='both',      
    left=False,
    right=True,    
    labelleft=False,
    labelright=True)  
    
    plt.yticks(np.arange(3)*50,np.arange(3)-1.0)
    
def go():
    
    calc_data_LA()
    plot_data(5,myLAtitle,'green',4,1)
    
    calc_data_T4()
    plot_data(4,myT4title,'red',1,5)
    
    calc_data_T5()
    plot_data(4,myT5title,'blue',6,5)
    
    plot_cbar()
    
go()
    
    

    
