# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:43:32 2024

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

fs_title  = 10
fs_label  = 3
fs_axis   = 8
fs_legend = 6

xsize = 0.3
ysize = 0.2

xstep = 0.4
ystep = 0.27

xoffs = 0.15
yoffs = 0.7

# ----------- H-Current ----------------------------------------

E_Ih          = +50.0  # in mV
Ih_midv       = -50.0
Ih_slope      = -0.25
tau_midv      = -50.0
Ih_gmax       = +50.0 
Ih_gain       = 0.0   # if set to 0, it will block Ih 

# ----------- K-Current ----------------------------------------

E_IK          = -90.0  # in mV
IK_midv       = -60.0
IK_slope      = +0.25
tauK          = 100.0
IK_gmax       = 10.0 
IK_gain       = 1.0   # if set to 0, it will block IK 

# --------------------------------------------------------------

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def plot_circuit():
    
    maxtime=2000
    nofcells=3
    
    g=np.zeros((nofcells,maxtime))
    
    tau       = 50.0
    synweight = 1
    
    myt=np.linspace(0,0.001*(maxtime-1),maxtime)
    vis_input = np.zeros((nofcells,maxtime))
    vis_input[0,500:1500] = 1.0
    
    #--------------------
    
    def set_connM():
        
        connM=np.zeros((nofcells,nofcells))
        
        connM[1,0] = synweight
        connM[2,1] = synweight
        connM[2,0] = synweight * (-1.1)
            
        return connM
    
           
    def calc_network():
        
        Vm=np.zeros((nofcells,maxtime))
        
        connM = set_connM() 
        
        for i in range(maxtime-1):
            
            time=i+1
            
            g[:,time]=connM.dot(Vm[:,time-1])+vis_input[:,time]
            Vm[:,time]=1/tau*(g[:,time]-Vm[:,time-1])+Vm[:,time-1]
            
        return Vm
    
    def finish_plot():
        
        plt.legend(loc=4,frameon=False,fontsize = fs_legend)
        
        plt.xticks(np.arange(5)*0.5,np.arange(5)*0.5)
        
        plt.ylim(-1.0,1.5)
        plt.xlim(0,2)
        
        plt.xlabel('time[s]',fontsize = fs_axis)
        plt.ylabel('response [au]',fontsize = fs_axis)
        
    mylw = 2
    
    Vm = calc_network()
    
    setmyaxes(xoffs , yoffs , xsize, ysize)
    
    plt.plot(myt,vis_input[0],linewidth=2,color='grey',label='input')
    plt.plot(myt,Vm[0],linewidth=2,color='red',label='1st order LP')
    plt.plot(myt,Vm[1],linewidth=2,color='blue',label='2nd order LP')
    finish_plot()
    
    setmyaxes(xoffs + xstep, yoffs , xsize, ysize)
    
    plt.plot(myt,2*Vm[2],linewidth=mylw,color='green')
    finish_plot()
        
def plot_H_parameters():
    
    mylw = 2
    
    Vm       = np.arange(100)-100
    Ih_ss    = 1.0/(1.0+np.exp((Ih_midv-Vm)*Ih_slope))
    tau      = 1.5/(np.exp(-0.1*(Vm-tau_midv))+np.exp(+0.1*(Vm-tau_midv)))+0.1

    setmyaxes(xoffs , yoffs - ystep, xsize, ysize)
    
    plt.plot(Vm,Ih_ss,label = 'Ih Activation',linewidth=mylw)
    plt.plot(Vm,tau,  label = 'Ih time const [s]',linewidth=mylw)
    plt.xlabel('membrane potential [mV]', fontsize = fs_axis)
    plt.legend(loc='upper right',frameon=False, fontsize = fs_legend)
    
    plt.xlim(-100,0)
    plt.ylim(-0.2,1.3)
    
def plot_K_parameters():
    
    mylw = 2
    
    Vm       = np.arange(100)-100
    Ih_ss    = 1.0/(1.0+np.exp((IK_midv-Vm)*IK_slope))
    tau      = Vm*0+tauK*0.001

    setmyaxes(xoffs , yoffs - 2*ystep, xsize, ysize)
    
    plt.plot(Vm,Ih_ss,label = 'IK Activation',linewidth=mylw)
    plt.plot(Vm,tau,  label = 'IK time const [s]',linewidth=mylw)
    plt.xlabel('membrane potential [mV]', fontsize = fs_axis)
    plt.legend(loc='upper right',frameon=False, fontsize = fs_legend)
    
    plt.xlim(-100,0)
    plt.ylim(-0.2,1.3)
    
def plot_recordings(Ih_gain, IK_gain):
    
    maxtime = 200
    deltat = 10.0
    g_leak = 2.0
    E_leak = -30.0
    capac  = +40.0 # in pF, results in 50ms membrane time-constant for g_leak = 1.0 nS
    cdt    = capac/deltat

    current_amp   = -50.0
    current_input = np.zeros(maxtime)
    current_input[50:150] = current_amp

    myt = np.arange(maxtime)*0.001*deltat

    def update_Vm(Vm,u,a,signal):

        Ih_ss   = 1.0/(1.0+np.exp((Ih_midv-Vm)*Ih_slope))
        tau     = 1.5/(np.exp(-0.1*(Vm-tau_midv))+np.exp(+0.1*(Vm-tau_midv)))*1000.0 + 100.0
        u       = deltat/tau*(Ih_ss-u)+u
        g_Ih    = u * Ih_gmax * Ih_gain
        
        IK_ss   = 1.0/(1.0+np.exp((IK_midv-Vm)*IK_slope))
        a       = deltat/tauK*(IK_ss-a)+a
        g_IK    = a * IK_gmax * IK_gain
        
        Vm = (g_leak*E_leak + E_Ih * g_Ih + E_IK * g_IK + cdt*Vm + signal) / (g_Ih + g_IK + g_leak + cdt)
        
        return Vm, u, a

    def calc_Vm():
        
        # all params need to be multi-columnar gain!
        
        u = 0
        a = 0
        
        Vm    = np.zeros(maxtime)
        Vm[0] = E_leak
        
        for t in range(1,maxtime): 
        
            Vm[t],u,a = update_Vm(Vm[t-1],u,a,current_input[t-1]) 
            
        Vm[0:50] = Vm[49]
            
        return Vm
    
    Vm = calc_Vm()
    
    if Ih_gain == 1:
    
        setmyaxes(xoffs + xstep, yoffs - ystep, xsize, ysize)
        
    if IK_gain == 1:
    
        setmyaxes(xoffs + xstep, yoffs - 2*ystep, xsize, ysize)
    
    plt.plot(myt, Vm,linewidth=2, color = 'green')
    plt.ylabel('membrane potential [mV]', fontsize = fs_axis)
    plt.xlabel('time [s]', fontsize = fs_axis)
    plt.xlim(0,2)
    
    if Ih_gain ==1: plt.ylim(-50,0)
    if IK_gain ==1: plt.ylim(-80,-50)


def plot_figure():
    
    plt.figure(figsize=(7.5,10))
    
    plot_circuit()
    plot_H_parameters()
    plot_K_parameters()
    plot_recordings(1,0) # with H-current
    plot_recordings(0,1) # with K-current
    
plot_figure()
    
    