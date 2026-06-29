# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 08:41:35 2024

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'
    
fs_legend = 8
fs_tick   = 5
fs_axis   = 8
fs_title  = 12

xsize = 0.3
ysize = 0.2

xstep = 0.45
ystep = 0.27

xoffs = 0.15
yoffs = 0.7

nofcells  = 65
ctype      = np.load('Circuits/ctype.npy')
ctype[18]  = 'CT1L'
ctype[19]  = 'CT1M'

cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

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

def read_data(Ih_gain):

    dirname = 'fig7_data/'

    if Ih_gain == 0: filelabel = 'no_Ih.npy'
    if Ih_gain == 1: filelabel = 'with_Ih.npy'
    
    all_costs  = np.load(dirname+'fig7_all_costs_'+filelabel)
    avg_params = np.load(dirname+'fig7_avg_params_'+filelabel)
    var_params = np.load(dirname+'fig7_var_params_'+filelabel)
    cov_m_allp = np.load(dirname+'fig7_cov_m_allp_'+filelabel)
    
    return all_costs, cov_m_allp, avg_params, var_params

def plot_figure():
    
    nofmodels = 10
    
    plt.figure(figsize=(7.5,10))

    bar_cmap = plt.get_cmap("plasma")
    
    for i in range(2):
        
        if i == 0: 
            
            all_costs, cov_m_allp, avg_params, var_params = read_data(0)
            
            avg_params[26:31] = 0
            var_params[26:31] = 0
            
        if i == 1: 
            
            all_costs, cov_m_allp, avg_params, var_params = read_data(1)
        
        # ----------------------------------------------------------------------
        
        setmyaxes(xoffs+i*xstep,yoffs-0.0*ystep,xsize,ysize)
        
        plt.bar(np.arange(nofmodels),all_costs,color = bar_cmap(np.arange(nofmodels)/(1.0*nofmodels)))
        plt.xticks(np.arange(nofmodels), np.arange(nofmodels)+1)
        plt.xlabel('model #',fontsize = fs_axis)
        plt.ylabel('cost [% of data power]',fontsize = fs_axis)
        plt.ylim(0,12)
        if i == 0: plt.title('no Ih', fontsize = fs_title)
        if i == 1: plt.title('with Ih', fontsize = fs_title)
        
        # ----------------------------------------------------------------------
        
        setmyaxes(xoffs+i*xstep,yoffs-1.0*ystep,xsize*1.1,ysize)
                
        plt.imshow(cov_m_allp,cmap = 'plasma',vmin=0)
        plt.xticks(np.arange(nofmodels), np.arange(nofmodels)+1)
        plt.yticks(np.arange(nofmodels), np.arange(nofmodels)+1)
        plt.xlabel('model #',fontsize = fs_axis)
        plt.ylabel('model #',fontsize = fs_axis)
        
        cbar = plt.colorbar()
        cbar.set_label('correlation', rotation=90,fontsize = fs_axis)
        
        # ----------------------------------------------------------------------
        
        setmyaxes(xoffs+i*xstep,yoffs-2.0*ystep,xsize,ysize)
        
        myscale = np.zeros(31)
        myscale[0:13]  = 0.3
        myscale[13:26] = 0.5
        myscale[26:31] = 0.8
        
        plt.bar(np.arange(31),avg_params,color=bar_cmap(myscale))
        
        for j in range(31):
            
            y1 = avg_params[j] - var_params[j]
            y2 = avg_params[j] + var_params[j]
            plt.plot([j,j],[y1,y2],color = bar_cmap(myscale[j]))
            
        plt.plot([12.5,12.5],[0,1],color='black',linewidth=0.5,linestyle='dashed')
        plt.plot([25.5,25.5],[0,1],color='black',linewidth=0.5,linestyle='dashed')
        
        plt.ylabel('mean normalized parameter values',fontsize = fs_axis)
        plt.xlabel('parameter #',fontsize = fs_axis)

        myxlabels = np.concatenate((cell_list,cell_list,cell_list[0:5]))
        
        plt.xticks(np.arange(31),myxlabels,rotation=90,fontsize=fs_tick)
        
        if i == 0: ymax = 0.25
        if i == 1: ymax = 1.0
            
        plt.text(2,0.85*ymax,'input gain',color=bar_cmap(0.3),fontsize = fs_legend)
        plt.text(15,0.85*ymax,'output gain',color=bar_cmap(0.5),fontsize = fs_legend)
        plt.text(25.7,0.85*ymax,'Ih gain',color=bar_cmap(0.8),fontsize = fs_legend)
        
        plt.xlim(-1,31)
        plt.ylim(0,ymax)
  
plot_figure()

    
    