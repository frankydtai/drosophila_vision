# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:36:29 2024

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=7)
plt.rc('ytick', labelsize=7)

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

fs_title = 10
fs_label = 4
fs_ticks = 6
fs_cbarl = 8

def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def draw_frame(mymin,mymax,mycolor):
     
    plt.plot([mymin, mymax],[mymin,mymin],color='black')
    plt.plot([mymin, mymax],[mymax,mymax],color='black')
    plt.plot([mymin,mymin],[mymin, mymax],color='black')
    plt.plot([mymax,mymax],[mymin, mymax],color='black')

def plot_figure():
    
    multi_colM = np.load('Circuits/multi_colM.npy')
    intra_colM = np.load('Circuits/intra_colM.npy')
    inter_colM = np.load('Circuits/inter_colM.npy')
    ctype      = np.load('Circuits/ctype.npy')
    ctype[18]  = 'CT1L'
    ctype[19]  = 'CT1M'
    
    mynofcells= multi_colM.shape[0]
    
    n = int(mynofcells/65)

    plt.figure(figsize=(7.5,10))
    
    xsize = 0.45
    ysize = 0.30
    
    # -------------------------------------------------

    setmyaxes(0.05,0.39,xsize,ysize)
    
    plt.imshow(intra_colM,vmin=-10,vmax=10,cmap='coolwarm',interpolation='None') 
    plt.xticks(np.arange(65), ctype, rotation=90, fontsize=fs_label)
    plt.yticks(np.arange(65), ctype, rotation=00, fontsize=fs_label)
    plt.title('intra-column connectivity',fontsize=fs_title,color='green')
    
    # -------------------------------------------------

    setmyaxes(0.53,0.39,xsize,ysize)
    
    plt.imshow(inter_colM,vmin=-10,vmax=10,cmap='coolwarm',interpolation='None') 
    plt.xticks(np.arange(65), ctype, rotation=90, fontsize=fs_label)
    plt.yticks(np.arange(65), ctype, rotation=00, fontsize=fs_label)
    plt.title('inter-column connectivity',fontsize=fs_title,color='orange')
    
    # -------------------------------------------------

    setmyaxes(0.15,0.02,xsize*1.5,ysize)
    
    plt.imshow(multi_colM,vmin=-10,vmax=10,cmap='coolwarm',interpolation='None') 
    plt.axis('off')
    plt.title('overall connectivity',fontsize=fs_title)
            
    # add outline
    
    def draw_square(x,y,mycolor):
        
        mylw  = 3
        myle  = 60
        
        plt.plot([x,x+myle],[y,y],color=mycolor,linewidth=mylw)
        plt.plot([x,x+myle],[y+myle,y+myle],color=mycolor,linewidth=mylw)
        plt.plot([x,x],[y,y+myle] ,color=mycolor,linewidth=mylw)
        plt.plot([x+myle,x+myle],[y,y+myle],color=mycolor,linewidth=mylw)
        
    
    myin = 3
    
    for i in range(n):
        
        # center
        
        x = i*65+myin
        y = i*65+myin
        
        draw_square(x,y,'green')
        
        #left
        
        if i > 0:
        
            x = (i-1)*65+myin
            y = i*65+myin
            
            draw_square(x,y,'orange')
        
        #right
        
        if i < 4:
        
            x = (i+1)*65+myin
            y = i*65+myin
            
            draw_square(x,y,'orange')
            
    # add grid
    
    for i in range(1,n):
            
        plt.plot([i*65,i*65],[0,n*65],color='black',linestyle='dashed')
        plt.plot([0,n*65],[i*65,i*65],color='black',linestyle='dashed')
        
    # frame it
    
    draw_frame(0,325,'black')
        
    plt.xlim(-1,326)
    plt.ylim(326,-1)
    
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fs_ticks)
    cbar.set_label('inhib      # of synapses      excit', rotation=90, fontsize=fs_cbarl)
    
plot_figure()