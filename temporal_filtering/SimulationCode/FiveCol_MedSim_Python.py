# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:53:25 2023

@author: aborst
"""
import numpy as np
import matplotlib.pyplot as plt
import blindschleiche_py3 as bs
import time

import Medulla_Library as ml

#################################################################
# Medulla Library contains:
# ml.read_ConnMs()
# ml.read_RecF_data(): RecF_data (13,45), ImpR_data (13,45)
# plot_ConnM(): Big ConnM + intra + inter M
# stimulus generation -> signal
#################################################################

from scipy.optimize import minimize

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

nofcells  = 65
nofcols   = 5
counter   = 0
maxtime   = 200
maxiter   = 500

# important model params

deltat    = 10.0  # in msec
g_leak    = 1.0   # in nS
E_exc     = +10.0 # in mV
E_inh     = -70.0 # in mV
capac     = +40.0 # in pF, results in 50ms membrane time-constant for g_leak = 1.0 nS
trld      = -50.0 # in mV: below trld, no signal is transmitted
cdt       = capac/deltat

Ca_tau    = 50.0  # in msec

E_leak    = np.zeros(nofcells*nofcols)-50.0

for i in range(nofcols):
    
    # Lamina cells L1-L3 get E_leak of - 20.0 mV
    
    E_leak[nofcells*i+8:nofcells*i+11]  = -20.0

exc_synweight = 0.001
inh_synweight = 0.001
ele_synweight = 0.000

# ----------- H-Current ----------------------------------------

E_Ih          = +50.0  # in mV
Ih_midv       = -50.0
Ih_slope      = -0.25
tau_midv      = -50.0
Ih_gmax       = +50.0 

Ih_gain       = 1.0   # if set to 0, it will block Ih 

signal_amp    = +40.0 # amplitude of current injection in photoreceptors, in pA.

g_total       = np.zeros((325,maxtime))

# parameter and cost function definition

nofparams = 2 * nofcells + 8 # 8 params for Ih (5 x gmax + 3)

# multicol_M, ctype

multi_colM = np.load('Circuits/multi_colM.npy')
ctype      = np.load('Circuits/ctype.npy')
ctype[18]  = 'CT1L'
ctype[19]  = 'CT1M'

multi_colM[130+11,0:130] = 0
multi_colM[130+11,195:325] = 0

# initialize with no cells clamped

clamped_cell_index      = []
clamped_cell_data_index = []

# -------------------------------------------------------------------------
# -------------- reading cell data and connectivity matrices --------------
# -------------------------------------------------------------------------

def init_network():
    
    # chemical synpases

    M_exc = exc_synweight * multi_colM * (multi_colM > 0)
    M_inh = inh_synweight * multi_colM * (multi_colM < 0) * (-1)
    
    signal = np.zeros((325,200))
    signal[130:138,50:200] = signal_amp

    data = ml.read_RecF_data()*20.0 # 20 mV Amplitude
    
    # ------------------------------
    
    return M_exc, M_inh, data, signal

M_exc, M_inh, data, signal = init_network()

def calc_multi_col_params(param):
    
    # turns params for one column into multi-column params
    
    multi_col_param=np.zeros(65*nofcols)
    
    for i in range(nofcols):
        
        multi_col_param[i*65:i*65+65] = param
        
    return multi_col_param

cost_array = np.zeros(maxiter*nofparams)

# -------------------------------------------------------------

cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])

def get_cell_index(mycell):
    
    index = np.zeros(13)
    
    for j in range(nofcells):
        if ctype[j]==mycell:
            index=j
            
    return index

def create_cell_index():
    
    cell_index = np.zeros(13)
    
    for i in range(13):
        
        cell_index[i] = get_cell_index(cell_list[i])
        
    return cell_index.astype(int)

cell_index = create_cell_index()

def clamp_cell(mycell):
    
    clamped_cell_index = 130 + get_cell_index(mycell)
    
    clamped_cell_data_index = np.where(cell_list == mycell)[0][0]
    
    return clamped_cell_index, clamped_cell_data_index
    
# ------- connectivity matrix ----------------------------------
    
def show_input_output(celltype,inorout='Input'):
    
    intra_colM = np.load('Circuits/intra_colM.npy')
    inter_colM = np.load('Circuits/inter_colM.npy')
    
    maxnofcells=20
    plt.figure(figsize=(5,8))
    myctype=get_cell_index(celltype)
    
    for j in range(2):
        
        plt.subplot(2,1,j+1)
        
        if j == 0: 
            M = intra_colM
            mytitle = celltype + ' Intra-columnar '+inorout
            
        if j == 1: 
            M = inter_colM
            mytitle = celltype + ' Inter-columnar '+inorout
            
        if inorout == 'Input':
        
            data=np.sort(np.abs(M[myctype]))
            args=np.argsort(np.abs(M[myctype]))
            signs=np.sign(M[myctype])
            
        if inorout == 'Output':
        
            data=np.sort(np.abs(M[:,myctype]))
            args=np.argsort(np.abs(M[:,myctype]))
            signs=np.sign(M[:,myctype])
        
        data=data[::-1]
        args=args[::-1]
        
        mycolor=['' for x in range(maxnofcells)]
        
        for i in range(maxnofcells):
            mycolor[i]='red'
            if signs[args[i]]<0: mycolor[i]='blue'
            
        xlabels=np.array(ctype)
        xlabels=xlabels[args[0:maxnofcells]]
            
        plt.bar(np.arange(maxnofcells)+1.0,data[0:maxnofcells],color=mycolor)
        plt.xticks(np.arange(maxnofcells)+1.0, xlabels, rotation=90, fontsize=6)
        plt.xlim(0,maxnofcells+1)
        
        plt.text(0.7 * maxnofcells, 0.9*np.max(data),'excitatory', color = 'red')
        plt.text(0.7 * maxnofcells, 0.8*np.max(data),'inhibitory', color = 'blue')
        
        plt.title(mytitle)
        plt.ylabel('number of synapses')
    
    print(celltype, np.sum(data[0:maxnofcells]),' total number of synapses')
    
    
# ------- plotting etc  ----------------------------------

def det_max_amp(x,myaxis):
    
    max_amp = np.max(x, axis = myaxis)
    min_amp = np.min(x, axis = myaxis)
    
    larger_max = (abs(max_amp) >= abs(min_amp))*1.0
    
    return max_amp * larger_max + min_amp * (1-larger_max)
     
def plot_model(data, model, label1 = 'data', label2 = 'model'):
    
    fontsize_legend = 8
    fontsize_ticklabels = 8
    fontsize_axislabel = 9
    
    mylw = 2
    
    # set x and y position for each cell type
    
    xpos = np.zeros(13)
    ypos = np.zeros(13)
    
    # L1-5
    
    xpos[0:5] = 0.25+np.arange(5)*0.11
    ypos[0:5] = 0.77
    
    # T4-Inputs
    
    xpos[5:9] = 0.06+np.arange(4)*0.11
    ypos[5:9] = 0.27
    
    # T5-Inputs
    
    xpos[9:13] = 0.55+np.arange(4)*0.11
    ypos[9:13] = 0.27
    
    xsize = 0.09
    ysize = 0.15
            
    def set_yticks():
        
        if i == 0 or i == 5 or i == 9:
            plt.yticks(np.arange(7)*10-30,np.arange(7)*10-30,fontsize=fontsize_ticklabels)
        else:
            plt.yticks(np.arange(7)*10-30,'')
            
    ylabelset = set([0,5,9])
            
    plt.figure(figsize=(16,9))
    
    for i in range(13):
                
        # Extract Impulse Responses from xt

        ImpR_model = 1.0*model[i,4]
        
        ImpR_data  = 1.0*data[i,4]
        
        # Extract Receptive field from xt
        
        maxamp_model = np.max(abs(ImpR_model))
        maxamp_data  = np.max(abs(ImpR_data))
        
        maxt_model = np.where(abs(ImpR_model) == np.max(abs(ImpR_model)))[0][0]
        maxt_data  = np.where(abs(ImpR_data)  == np.max(abs(ImpR_data)))[0][0]
        
        RecF_model = bs.rebin(model[i,:,maxt_model],45)
        RecF_model = bs.blurr(RecF_model,5)
        RecF_model = RecF_model/np.max(abs(RecF_model))*maxamp_model
        
        RecF_data  = bs.rebin(data[i,:,maxt_data],45)
        RecF_data  = bs.blurr(RecF_data,5)
        RecF_data  = RecF_data/np.max(abs(RecF_data))*maxamp_data
        
        # --------plotting ---------------------------------------
        
        bs.setmyaxes(xpos[i],ypos[i],xsize,ysize)
        
        plt.plot(np.roll(RecF_data,-2),color='gray',label=label1,linewidth = mylw)
        plt.plot(np.roll(RecF_model,-2),color='red',label=label2,linewidth = mylw)
            
        plt.ylim(-30,30)
        set_yticks()
        
        plt.xlim(0,40)
        
        plt.xticks(np.arange(5)*10,np.arange(5)*10-20,fontsize=fontsize_ticklabels)
        if i in [0,5,9]: plt.legend(loc=1,frameon=False,fontsize=fontsize_legend)
        plt.title(cell_list[i])
            
        plt.xlabel('visual angle [deg]',fontsize=fontsize_axislabel)
        
        if i in ylabelset: plt.ylabel('response [mV]')
                
        bs.setmyaxes(xpos[i],ypos[i]-0.20,xsize,ysize)
        
        plt.plot(ImpR_data,color='gray',label=label1,linewidth = mylw)
        plt.plot(ImpR_model,color='red',label=label2,linewidth = mylw)
        
        cost = np.sum((ImpR_model - ImpR_data)**2)/np.sum((ImpR_data)**2)
        match = int(100*(1-cost))
        
        plt.text(95,-26,'match: ' + str(match) + ' %',fontsize=fontsize_legend)
            
        plt.ylim(-30,30)
        set_yticks()
        
        plt.xlim(0,200)
        
        plt.xticks(np.arange(5)*50,np.arange(5)*0.5,fontsize=fontsize_ticklabels)
            
        plt.xlabel('time [s]',fontsize=fontsize_axislabel)
        if i in ylabelset: plt.ylabel('response [mV]')
        
        if i == 0:
            plt.text(530,130,'Lamina',fontsize=14)
            
        if i == 5:
            plt.text(405,130,'T4 Input',fontsize=14)
            
        if i == 9:
            plt.text(405,130,'T5 Input',fontsize=14)
        
        plt.pause(0.1)
        
def plot_many_models(data,model):
    
    nofmodels = model.shape[0]
    
    fontsize_legend = 8
    fontsize_ticklabels = 8
    fontsize_axislabel = 9
    
    mylw = 2
    
    # set x and y position for each cell type
    
    xpos = np.zeros(13)
    ypos = np.zeros(13)
    
    # L1-5
    
    xpos[0:5] = 0.25+np.arange(5)*0.11
    ypos[0:5] = 0.77
    
    # T4-Inputs
    
    xpos[5:9] = 0.06+np.arange(4)*0.11
    ypos[5:9] = 0.27
    
    # T5-Inputs
    
    xpos[9:13] = 0.55+np.arange(4)*0.11
    ypos[9:13] = 0.27
    
    xsize = 0.09
    ysize = 0.15
            
    def set_yticks():
        
        if i == 0 or i == 5 or i == 9:
            plt.yticks(np.arange(7)*10-30,np.arange(7)*10-30,fontsize=fontsize_ticklabels)
        else:
            plt.yticks(np.arange(7)*10-30,'')
            
    ylabelset = set([0,5,9])
            
    plt.figure(figsize=(16,9))
    
    for i in range(13):
        
        # --------------- upper panel -------------------------------
        
        bs.setmyaxes(xpos[i],ypos[i],xsize,ysize)
        
        ImpR_data  = 1.0*data[i,4]
        maxt_data  = np.where(abs(ImpR_data)  == np.max(abs(ImpR_data)))[0][0]
        RecF_data  = bs.rebin(data[i,:,maxt_data],45)
        RecF_data  = bs.blurr(RecF_data,5)
        
        plt.plot(np.roll(RecF_data,-2),color='gray',label='data',linewidth = mylw)
        
        plt.title(cell_list[i])
        plt.xlabel('visual angle [deg]',fontsize=fontsize_axislabel)
        if i in ylabelset: plt.ylabel('response [mV]')
        
        plt.ylim(-30,30)
        set_yticks()     
        plt.xlim(0,40)
        plt.xticks(np.arange(5)*10,np.arange(5)*10-20,fontsize=fontsize_ticklabels)
        if i in ylabelset: plt.legend(loc=1,frameon=False,fontsize=fontsize_legend)
        
        for k in range(nofmodels):
                
            # Extract Impulse Responses from xt
    
            ImpR_model = 1.0*model[k,i,4]
            
            # Extract Receptive field from xt
            
            maxt_model = np.where(abs(ImpR_model) == np.max(abs(ImpR_model)))[0][0]      
            RecF_model = bs.rebin(model[k,i,:,maxt_model],45)
            RecF_model = bs.blurr(RecF_model,5)
            
            plt.plot(np.roll(RecF_model,-2),color='red',linewidth = 1)
            
        # ---------- lower panel ---------------------
                
        bs.setmyaxes(xpos[i],ypos[i]-0.20,xsize,ysize)
        
        ImpR_data  = 1.0*data[i,4]
        maxt_data  = np.where(abs(ImpR_data)  == np.max(abs(ImpR_data)))[0][0]
        RecF_data  = bs.rebin(data[i,:,maxt_data],45)
        RecF_data  = bs.blurr(RecF_data,5)
        
        plt.plot(ImpR_data,color='gray',label=data,linewidth = mylw)
        
        plt.ylim(-30,30)
        set_yticks()
        plt.xlim(0,200)
        plt.xticks(np.arange(5)*50,np.arange(5)*0.5,fontsize=fontsize_ticklabels)
        plt.xlabel('time [s]',fontsize=fontsize_axislabel)
        if i in ylabelset: plt.ylabel('response [mV]')
        
        if i == 0:
            plt.text(530,130,'Lamina',fontsize=14)
            
        if i == 5:
            plt.text(405,130,'T4 Input',fontsize=14)
            
        if i == 9:
            plt.text(405,130,'T5 Input',fontsize=14)
        
        for k in range(nofmodels):
    
            ImpR_model = 1.0*model[k,i,4]
            
            # Extract Receptive field from xt
            
            maxt_model = np.where(abs(ImpR_model) == np.max(abs(ImpR_model)))[0][0] 
            RecF_model = bs.rebin(model[k,i,:,maxt_model],45)
            RecF_model = bs.blurr(RecF_model,5)

            plt.plot(ImpR_model,color='red',linewidth = 1)
        
def plot_all_responses(all_responses,index):
    
    # here, index refers to the cell_list ['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9']
    # all_responses are calculated by big_stimulus_loop(z)
    
    fontsize_legend = 6
    fontsize_axislabel = 10
    
    plt.figure(figsize=(14,9))
    
    myt     = np.arange(1000)*0.01
    mylw    = 2
    
    stim_list = ['Grating','ON Edge','OFF Edge','ON Bar','OFF Bar']
    velo_list = [15,30,60,120]
    
    mymax = np.max(all_responses[index])+10
    mymin = np.min(all_responses[index])-10
    
    mymax = +15.0
    mymin = -65.0
    
    for i in range(4):
        
        for j in range(5):
            
            plt.subplot(4,5,i*5+j+1)
            
            mylabel = 'R1-6 signal ' + str(velo_list[i]) +' deg/s'
            
            plt.plot(myt,all_responses[index,j,i],linewidth=mylw,color='red')
            plt.plot(myt,all_responses[13,j,i],color='gray',label=mylabel)
            plt.legend(loc=1,frameon=False,fontsize=fontsize_legend)
            
            plt.xlim(2,8)
            plt.ylim(mymin,mymax)
            
            if j == 0:
                
                plt.ylabel('membr. potential [mV]',fontsize=fontsize_axislabel)
            
            if i == 0: 
                
                plt.title(cell_list[index] + ' ' + stim_list[j])
                
            if i == 3:
                
                plt.xlabel('time [s]',fontsize=fontsize_axislabel)
    
def plot_params(z,all_cells = 0,mytitle =''):
    
    plt.figure(figsize=(7,11))
    
    fontsize_legend = 8
    xpos = -20
    mylw = 3
    
    if all_cells == 1:
        
        plot_index = np.arange(nofcells)
        plot_list  = ctype
        
    else:
            
        plot_index = cell_index
        plot_list  = cell_list
        
    max_num = plot_index.shape[0]
    
    my_cmap = plt.get_cmap("viridis")
    
    for i in range(2):
    
        plt.subplot(3,1,i+1)
        
        plt.bar(np.arange(max_num),z[plot_index+i*65],color=my_cmap(np.arange(max_num)/(1.0*max_num)))
        
        if all_cells == 0:
            
            plt.xticks(np.arange(max_num),plot_list)
            
        else:
            
            plt.xticks(np.arange(max_num),plot_list,rotation='vertical',fontsize=6)    
        
        if i == 0: 
            
            plt.ylabel('input gain')
            plt.title(mytitle)
            
        if i == 1: plt.ylabel('output gain')
        
        plt.yscale('log')
        plt.ylim(0.05,500)
    
    plt.subplot(3,1,3)
    
    Ih_gmax  = z[130:135]
    Ih_midv  = z[135]
    Ih_slope = z[136]
    tau_midv = z[137]
    
    Vm       = np.arange(100)-100
    Ih_ss    = 1.0/(1.0+np.exp((Ih_midv-Vm)*Ih_slope))
    tau      = 1.5/(np.exp(-0.1*(Vm-tau_midv))+np.exp(+0.1*(Vm-tau_midv)))+0.1
    
    plt.plot(Vm,Ih_ss,label = 'Ih Activation',linewidth=mylw)
    plt.plot(Vm,tau,  label = 'Ih time constant [s]',linewidth=mylw)
    plt.xlabel('membrane potential [mV]')
    plt.legend(loc=1,frameon=False, fontsize = fontsize_legend)
    
    plt.text(xpos,0.7,'Ih_midv  = ' +str(int(Ih_midv*100)/100.0),fontsize = fontsize_legend)
    plt.text(xpos,0.6,'Ih_slope = ' +str(int(Ih_slope*100)/100.0),fontsize = fontsize_legend)
    plt.text(xpos,0.5,'tau_midv = ' +str(int(tau_midv*100)/100.0),fontsize = fontsize_legend)
    
    bs.setmyaxes(0.2,0.2,0.2,0.1)
    plt.bar(np.arange(5),Ih_gmax)
    plt.xticks(np.arange(5),['L1','L2','L3','L4','L5'],fontsize = fontsize_legend)
    plt.yticks(np.arange(5)*20,np.arange(5)*20,fontsize = fontsize_legend)
    plt.title('Ih_gmax',fontsize = fontsize_legend)
    
# ------- network calculations  -----------------------------------------------

def rectsyn(x,thrld):
    
    result=x-thrld
    result=result*(result>0)
    
    return result

def update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,signal):

    Ih_ss   = 1.0/(1.0+np.exp((Ih_midv-Vm)*Ih_slope))
    tau     = 1.5/(np.exp(-0.1*(Vm-tau_midv))+np.exp(+0.1*(Vm-tau_midv)))*1000.0 + 100.0
    u       = deltat/tau*(Ih_ss-u)+u
    g_Ih    = u * Ih_gmax * Ih_gain
    
    g_exc   = np.dot(M_exc,(rectsyn(Vm,trld)*out_gain))*inp_gain
    g_inh   = np.dot(M_inh,(rectsyn(Vm,trld)*out_gain))*inp_gain
    
    Vm = (g_exc*E_exc + g_inh*E_inh + g_leak*E_leak + E_Ih * g_Ih + cdt*Vm + signal)
    Vm = Vm / (g_exc + g_inh + g_Ih + g_leak + cdt)
    
    return Vm, u

def calc_network(cell_input,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv):
    
    # all params need to be multi-columnar gain!
    
    maxtime = cell_input.shape[1]
    
    u       = np.zeros(325)
    
    Vm      = np.zeros((325,maxtime))
    Vm[:,0] = E_leak
    
    for t in range(1,maxtime): 
    
        Vm[:,t],u = update_Vm(Vm[:,t-1],u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,cell_input[:,t-1])
        
        Vm[clamped_cell_index,t] = E_leak[clamped_cell_index] + data[clamped_cell_data_index,4,t % 200]
        
    return Vm

def calc_linear_network(cell_input,inp_gain,out_gain):
    
    # all params need to be multi-columnar gain!
    
    n       = M_exc.shape[0]
    
    tau     = 50.0
    
    maxtime = cell_input.shape[1]
    
    Vm      = np.zeros((n,maxtime))
    Vm[:,0] = E_leak
    
    M       = M_exc - M_inh
    
    for t in range(1,maxtime): 
        
        net_inp = M.dot(Vm[:,t-1] * out_gain) * inp_gain + cell_input[:,t-1]
        Vm[:,t] = 1.0/(tau/deltat) * (net_inp - Vm[:,t-1]) + Vm[:,t-1]
        
    Vm[:,0:50] = 0
        
    return Vm
        
def calc_eigen():
    
    intra_colM = np.load('Circuits/intra_colM.npy')
    inter_colM = np.load('Circuits/inter_colM.npy')
    
    plt.figure(figsize=(15,11))
    
    for i in range(1,4):
    
        if i == 1: 
            connM   = intra_colM
            mytitle = 'intra columnar'
        if i == 2: 
            connM = inter_colM
            mytitle = 'inter columnar'
        if i == 3: 
            connM = intra_colM + inter_colM
            mytitle = 'intra + inter'
        
        eigenvals,eigenvecs = np.linalg.eig(connM)
        
        eigenvals      = np.sqrt(np.real(eigenvals)**2+np.imag(eigenvals)**2)
        eigenvecs      = np.sqrt(np.real(eigenvecs)**2+np.imag(eigenvecs)**2)
        
        sortargs = np.argsort(eigenvals)[::-1]
    
        plt.subplot(2,3,i)
        plt.plot(eigenvals[sortargs])
        plt.ylabel('EigenValues')
        plt.title(mytitle)
        plt.subplot(2,3,i+3)
        plt.imshow(eigenvecs[:,sortargs],interpolation='None')
        plt.title('EigenMatrix')  

# ------- calculating Receptive Fields  --------------------------------------

def assign_params(z):

    inp_gain = calc_multi_col_params(z[0:65])
    out_gain = calc_multi_col_params(z[65:130])
    
    interim       = np.zeros(65)
    interim[8:13] = z[130:135]
    Ih_gmax       = calc_multi_col_params(interim)
    
    Ih_midv  = z[135]
    Ih_slope = z[136]
    tau_midv = z[137]
    
    return inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv

def calc_model(z):
    
    inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv = assign_params(z)
    
    model = np.zeros((13,9,200))
        
    # the stimulus excites central column photoreceptors
        
    Vm = calc_network(signal,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
        
    # columns 0,1,2,3,4 placed into xt_model 2,3,4,5,6
    
    for i in range(5):
        
        model[:,i+2]=Vm[i*65+cell_index]
        
    # removes DC value before stimulation
    
    interim = np.transpose(model,axes=(2,0,1)) - model[:,:,49]
    
    model  = np.transpose(interim,axes=(1,2,0))
    
    model[:,:,0:50] = 0
    
    # accounts for Ca-buffering
        
    model  = bs.lowpass(model,Ca_tau/deltat)
    
    # shift backwards one time pointm but leaves last point
    
    model[:,:,0:199] = model[:,:,1:200]
    
    return model

# -------- responses to stimulus sets -----------------------------------------

def calc_all_responses(z):
    
    velo_array=[15,30,60,120]
    
    all_responses = np.zeros((14,5,4,1000))
    
    # dim1: 13 cells plus R1
    # dim2: 5 stimulus conditions: grating, On edges, Off edges, On bars, Off bars
    # dim3: 4 velocities
    # dim4: 10 s time
    
    inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv = assign_params(z)
    
    for i in range(4):
        
        print(velo_array[i],' deg/s')
        
        for j in range(5):
            
            if j == 0: signal = ml.calc_grating(velo_array[i])
            if j == 1: signal = ml.calc_edge(velo_array[i], polarity='bright')
            if j == 2: signal = ml.calc_edge(velo_array[i], polarity='dark')
            if j == 3: signal = ml.calc_bar(velo_array[i], polarity='bright')
            if j == 4: signal = ml.calc_bar(velo_array[i], polarity='dark')
            
            Vm = calc_network(signal_amp*signal,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
            
            for k in range(13):
                
                all_responses[k,j,i] = Vm[4*65+cell_index[k]]
                
            # take photoreceptor output as reference signal
                
            all_responses[13,j,i] = Vm[4*65]
                  
    return all_responses   

def calc_chirp_responses(z,loc_global = 'global'):

    signal = ml.calc_chirp(loc_global=loc_global)
    
    inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv = assign_params(z)

    Vm = calc_network(signal_amp*signal,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
    
    chirp_responses = np.zeros((14,1000))
    
    for k in range(13):
        
        chirp_responses[k] = Vm[4*65+cell_index[k]]
        
    # take photoreceptor output as reference signal
        
    chirp_responses[13] = Vm[4*65]
    
    return chirp_responses

    
def show_cells(z,select=1):
    
    inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv = assign_params(z)
    
    Vm = calc_network(signal,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
    
    Vm = (np.transpose(Vm) - Vm[:,49]).transpose()
        
    plt.figure(figsize=(8,4))
    
    if select == 0:
        
        plt.imshow(np.transpose(Vm),vmin=-30,vmax=+30,cmap = 'coolwarm')
        
        for i in range(1,5):
            
            plt.plot([i*65,i*65],[0,200],linestyle='dashed',color='black',linewidth=0.5)
        
        plt.xticks(np.arange(5)*65+32,['col -2','col -1','center col','col +1','col +2'])
        plt.yticks(np.arange(5)*50,np.arange(5)*0.5)
        plt.ylim(200,0)
        plt.ylabel('time [sec]')
        
    if select == 1:
        
        plt.imshow(bs.rebin(Vm[130+cell_index],130,200),vmin=-40,vmax=+40,cmap = 'coolwarm')
        plt.yticks(np.arange(13)*10+5, cell_list)
        plt.xticks(np.arange(5)*50,np.arange(5)*0.5)
        plt.xlabel('time [sec]')
    
    cbar = plt.colorbar()
    cbar.set_label('response [mV]', rotation=90)
    
# ------------------ blocking experiments -------------------------------------

def block_output(celltype):
    
    myctype = get_cell_index(celltype)
    
    # set output gain of celltype to 0
    
    block_z = 1.0*z

    block_z[65+myctype] = 0
    
    model = calc_model(block_z)
    
    return model

def calc_MVP():
    
    all_costs = np.zeros(13)
    
    for i in range(13):
        
        block_z = 1.0*z

        block_z[65+cell_index[i]] = 0
        
        all_costs[i] = np.log10(calc_cost(block_z))
        
    sort_index = np.argsort(all_costs)[::-1]
    
    cost_matrix = np.zeros((13,13))
    
    for i in range(13):
        
        block_z = 1.0*z
        block_z[65+cell_index[i]] = 0
        
        model = calc_model(block_z)
        
        for j in range(13):
            
            cost_matrix[i,j] = np.sum((model[j]-data[j])**2)/np.sum((data[j])**2)*100.0  
            
    # --------- plotting ----------------------
    
    plt.figure(figsize=(14,5))
    
    mycmap = 'viridis'
    
    bar_cmap = plt.get_cmap(mycmap)
    
    plt.subplot(1,2,1)
    
    plt.bar(np.arange(13),all_costs[sort_index],color=bar_cmap(1-np.arange(13)/(13.0)))
    plt.xticks(np.arange(13),ctype[cell_index[sort_index]],rotation = 90)
    plt.ylabel('log10(cost [%])')
    plt.title('Overall Cost, determined by blocking')
      
    plt.subplot(1,2,2)     
    
    plt.imshow(np.log10(cost_matrix),cmap = mycmap)
    plt.yticks(np.arange(13),ctype[cell_index])
    plt.xticks(np.arange(13),ctype[cell_index],rotation = 90)
    plt.ylabel('blocked cell')
    plt.title('Cell-Specific Cost, determined by blocking')
    
    cbar = plt.colorbar()
    cbar.set_label('log10(cost [%])', rotation=90)
    
    
def block_connection(precell,postcell):
    
    global M_exc, M_inh, multi_colM
    
    my_precell  = get_cell_index(precell)
    my_postcell = get_cell_index(postcell)
    
    for i in range(nofcols):
        
        for j in range(nofcols):
    
            multi_colM[i*65+my_postcell,j*65+my_precell] = 0
        
    M_exc = exc_synweight * multi_colM * (multi_colM > 0)
    M_inh = inh_synweight * multi_colM * (multi_colM < 0) * (-1)
    
    model = calc_model(z)
    
    # reload original connectivity matrix
    
    multi_colM = np.load('Circuits/multi_colM.npy')
    
    M_exc = exc_synweight * multi_colM * (multi_colM > 0)
    M_inh = inh_synweight * multi_colM * (multi_colM < 0) * (-1)
    
    return model

def calc_comparison(z,mycell,curr_amp,block = 0):
    
    mylw = 3
    
    inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv = assign_params(z)
    
    my_index = get_cell_index(mycell)
    
    # ----------- visual stimulation --------------------
    
    visual = np.zeros((325,200))
    visual[130:138,50:150] = signal_amp
    
    Vm = calc_network(visual,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
    
    visual_response = Vm[130+my_index]
    
    # ------------current stimulation ---------------------
    
    current = np.zeros((325,200))
    current[130+my_index,50:150] = curr_amp
    
    if block == 1: out_gain[130+my_index] = 0
    
    Vm = calc_network(current,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
    
    current_response = Vm[130+my_index]
    
    plt.figure()
    
    plt.plot(visual_response, label = 'Visual',linewidth = mylw)
    plt.plot(current_response,label = 'Current',linewidth = mylw)
    
    plt.legend(loc=1,frameon=False)
    
    plt.xticks(np.arange(5)*50,np.arange(5)*0.5)
    plt.xlim(0,200)
    
    plt.ylabel('membrane potential [mV]')
    plt.xlabel('time [sec]')
    plt.title(mycell)
    
def plot_H_current(z,mycell):
    
    mylw = 2
    
    my_cmap = plt.get_cmap("viridis")
    
    inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv = assign_params(z)
    
    my_index = get_cell_index(mycell)
    
    #inp_gain[130+my_index] = 0
    out_gain[130+my_index] = 0
    
    recordings = np.zeros((10,200))
    
    plt.figure()
    
    # ------------current stimulation ---------------------
    
    for i in range(5):
    
        current  = np.zeros((325,200))
        curr_amp = -(2*i+2)*100
        current[130+my_index,50:150] = curr_amp
        
        Vm = calc_network(current,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv)
        
        recordings[i] = Vm[130+my_index]
        
        Vm[130+my_index,0:50] = Vm[130+my_index,49]
        
        plt.plot(Vm[130+my_index],label = str(curr_amp)+' pA',linewidth = mylw, color = my_cmap(i/5.0))

    plt.legend(loc=4,frameon=False)
    plt.xticks(np.arange(5)*50,np.arange(5)*0.5)
    plt.xlim(0,200)
    
    plt.ylabel('membrane potential [mV]')
    plt.xlabel('time [sec]')
    plt.title(mycell)
    
    #np.save('Models/L1_Currinj.npy', recordings)

# -----------------------------------------------------------------------------
# -------------- cost calculation and parameter search ------------------------
# -----------------------------------------------------------------------------

def calc_cost(z):
    
    global counter
    
    model = calc_model(z)
    
    cost  = np.sum((model-data)**2)
    power = np.sum(data**2)
    
    cost = cost/power*100.0
    
    cost_array[counter] = cost
    
    if (counter % nofparams == 0):
        
        print('run ', counter/nofparams, ' cost =', format(cost,'.4f'), '% of data power')
    
    counter += 1
        
    return cost

def test_time(z,nofruns=nofparams):

    a=time.time()
    
    for i in range(nofruns):
        
        calc_cost(z)
        
    b=time.time()
    
    print('total time    :', (b-a)*1000.0, ' msec')
    print('time per cycle:', (b-a)*1000.0/nofruns, ' msec')

def plot_cost_array(cost):
    
    plt.figure()
    plt.plot(cost)
    plt.yscale('log')
    
def calc_z_bounds():
    
    # input gain -----------------------------
      
    l_z_bound     = 0.1
    h_z_bound     = 100.0
    
    z_bounds = [(l_z_bound,h_z_bound)]
    
    for i in range(nofcells-1): 
        z_bounds.append((l_z_bound,h_z_bound))

    # output gain ----------------------------
        
    l_z_bound     = 0.1
    h_z_bound     = 100.0
    
    for i in range(nofcells): 
        z_bounds.append((l_z_bound,h_z_bound))
        
    # Ih_gmax
    
    for i in range(5):
    
        z_bounds.append((0,100))
    
    # Ih_midv
    
    z_bounds.append((-70,-30))
    
    # Ih_slope
    
    z_bounds.append((-0.40,-0.20))
    
    # tau_midv
    
    z_bounds.append((-70,-40))
    
    return z_bounds

z_bounds = calc_z_bounds()
    
def create_rand_params():
    
    z = np.zeros(nofparams)
    
    for i in range(nofparams):
            
        z_mean  = (z_bounds[i][1]+z_bounds[i][0])/2.0
        z_range = (z_bounds[i][1]-z_bounds[i][0])/2.0
        z[i]    = z_mean+(np.random.rand()-0.5) * z_range
            
    return z
    
def guess_initial_params():
    
    z = np.zeros(nofparams)
    
    z[0:65]   = + 0.5    + (np.random.rand(65)-0.5)*0.2
    z[65:130] = + 0.5    + (np.random.rand(65)-0.5)*0.2
    z[130:135] = Ih_gmax  + (np.random.rand(5)-0.5)*10.0
    z[132]     = 0.0                                        #L3
    z[135]     = Ih_midv  + (np.random.rand()-0.5)*5.0
    z[136]     = Ih_slope + (np.random.rand()-0.5)*0.02
    z[137]     = tau_midv + (np.random.rand()-0.5)*5.0
    
    return z
    
def calc_z_init(z,init_option):
    
    if init_option == 0:
        z = 1.0 * z   
    if init_option == 1:
        z=guess_initial_params()      
    if init_option == 2:
        z=create_rand_params()
        
    return z

def send_message(res,a,b):
    
    run_time = (b-a)/60.0
    
    print()
    print('Optimization Success  :', res.success)
    print('Last Value of cost fct:', format(res.fun,'.2f'))
    print('Number of cost fct use:', res.nfev)
    print('total run time        :', format(run_time, '.2f'), ' min')
    print()
       
def fit_params(z, init_option, plotit = 1):
    
    global counter
    
    a=time.time()
    counter = 0 
    z_init  = calc_z_init(z,init_option)
    options   = {'maxiter':maxiter*nofparams}
            
    res = minimize(calc_cost, z_init, method = 'L-BFGS-B', tol = 1e-8, bounds = z_bounds, options = options)
   
    z = res.x

    b=time.time()
    send_message(res,a,b)
    
    if plotit == 1:
    
        plot_cost_array(cost_array[0:counter]) 
        plot_model(data,calc_model(z))
        plot_params(z)
    
    return z

def do_many_runs(z,nofruns,fname):
    
    dirname = 'Parameter/'
    
    all_params = np.zeros((nofruns,nofparams))
    
    for i in range(nofruns):
        
        print()
        print('round',i)
        print()
        
        all_params[i] = fit_params(z,1,plotit=0)
        
        np.save(dirname+fname,all_params)
    
def eval_many_runs(fname):
    
    dirname = 'Parameter/'
    
    all_params  = np.load(dirname + fname)
    nofruns     = all_params.shape[0]
    all_costs   = np.zeros(nofruns)
    norm_params  = np.zeros((nofruns,nofparams))
    
    for i in range(nofruns):
        
        all_costs[i] = calc_cost(all_params[i])
        
    sort_indx  = np.argsort(all_costs)
    
    all_costs  = all_costs[sort_indx]
    all_params = all_params[sort_indx]
    
    norm_params = all_params - np.min(all_params,axis=0)
    norm_params = norm_params/np.max(norm_params,axis=0)*100.0
        
    plt.figure(figsize=(7.5,10)) 
    
    plt.subplot(2,1,1)
    
    plt.plot(all_costs,np.arange(nofruns),linewidth = 2, color = 'red')
    plt.ylim(nofruns,0)
    plt.xlabel('cost')
    plt.ylabel('sorted run')
    
    plt.subplot(2,1,2)
    
    plt.imshow(bs.rebin(norm_params,100,nofparams))
    plt.xlabel('parameter')
    plt.ylabel('sorted run')   
    cbar = plt.colorbar()
    cbar.set_label('normalized parameter value', rotation=90)
    
    return all_params

def eval_diff_models(all_params, noftopmodels = 10, mycmap = 'plasma'):
    
    nofmodels  = all_params.shape[0]
    all_costs  = np.zeros(nofmodels)
    many_models = np.zeros((nofmodels,13,9,200))
    
    for i in range(nofmodels): 

        all_costs[i]  = calc_cost(all_params[i])
        print(i,format(all_costs[i],'.2f'))
        many_models[i] = calc_model(all_params[i])
        
    sort_indx  = np.argsort(all_costs)

    all_costs  = all_costs[sort_indx]
    all_params = all_params[sort_indx]
    
    # take only top models
    
    nofmodels = noftopmodels
    
    all_params = all_params[0:nofmodels,:]
    nrm_params = np.zeros((nofmodels,nofparams))
    all_costs  = all_costs[0:nofmodels]
    many_models = many_models[0:nofmodels]
            
    # variance across normalized parameters
    
    for i in range(nofparams):
    
        nrm_params[:,i] = all_params[:,i] - z_bounds[i][0]
        nrm_params[:,i] = nrm_params[:,i] / (z_bounds[i][1]-z_bounds[i][0])
        
    # correlation of parameter sets
    
    allp_index = np.arange(135)
    selp_index = np.concatenate((cell_index,cell_index+65,np.arange(5)+130))
            
    cov_m_allp = np.corrcoef(all_params[:,allp_index])
    cov_m_selp = np.corrcoef(all_params[:,selp_index])
            
    avg_params = np.mean(nrm_params,axis=0)[selp_index]
    var_params = np.var(nrm_params,axis=0)[selp_index]
    
    dirname = 'fig7_data/'
    
    if Ih_gain == 0: filelabel = 'no_Ih.npy'
    if Ih_gain == 1: filelabel = 'with_Ih.npy'
    
    np.save(dirname+'fig7_all_costs_'+filelabel,all_costs)
    np.save(dirname+'fig7_avg_params_'+filelabel,avg_params)
    np.save(dirname+'fig7_var_params_'+filelabel,var_params)
    np.save(dirname+'fig7_cov_m_selp_'+filelabel,cov_m_selp)
    np.save(dirname+'fig7_cov_m_allp_'+filelabel,cov_m_allp)
            
    plt.figure(figsize=(12,9))
    
    bar_cmap = plt.get_cmap("viridis")
    
    plt.subplot(2,2,1)
    
    plt.bar(np.arange(nofmodels),all_costs,color = bar_cmap(np.arange(nofmodels)/(1.0*nofmodels)))
    plt.xticks(np.arange(nofmodels), np.arange(nofmodels)+1)
    plt.xlabel('model #')
    plt.ylabel('cost [% of data power]')
    
    plt.subplot(2,2,2)
            
    plt.imshow(cov_m_allp,cmap = mycmap,vmin=0)
    plt.xticks(np.arange(nofmodels), np.arange(nofmodels)+1)
    plt.yticks(np.arange(nofmodels), np.arange(nofmodels)+1)
    plt.xlabel('model #')
    plt.ylabel('model #')
    plt.title('Covariance All Parameters')
    
    cbar = plt.colorbar()
    cbar.set_label('correlation', rotation=90)
    
    plt.subplot(2,2,3)
    
    plt.fill_between(np.arange(31),avg_params-var_params,avg_params+var_params,color='blue')
    #plt.plot(avg_params,color='black',linewidth=2)
    
    plt.ylabel('mean +- var of normalized param values')
    plt.xlabel('parameter #')
    plt.xlim(-1,31)
    
    plt.subplot(2,2,4)
            
    plt.imshow(cov_m_selp,cmap = mycmap,vmin=0)
    plt.xticks(np.arange(nofmodels), np.arange(nofmodels)+1)
    plt.yticks(np.arange(nofmodels), np.arange(nofmodels)+1)
    plt.xlabel('model #')
    plt.ylabel('model #')
    plt.title('Covariance C-Type Parameters')
    
    cbar = plt.colorbar()
    cbar.set_label('correlation', rotation=90)
    
    #plot_many_models(data,many_models)
    
z = np.load('FiveCol_Parameter/with_Ih/best_parameter.npy')

calc_cost(z)
model = calc_model(z)

all_params = np.load('FiveCol_Parameter/with_Ih/3rdround_paramsets.npy')
eval_diff_models(all_params, noftopmodels = 10, mycmap = 'plasma')






    



