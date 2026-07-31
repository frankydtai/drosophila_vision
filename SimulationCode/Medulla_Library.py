# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 08:43:49 2018

@author: aborst
"""

import numpy as np
import matplotlib.pyplot as plt
import blindschleiche_py3 as bs
from scipy.signal import chirp

dirname='Connectivity Matrix Srini/'
fname1='central_column_connectivity.csv'
fname2='offset_column_connectivity.csv'
    
nofcells = 65
nofcols  = 5

cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])

ctype      = np.load('Circuits/ctype.npy')


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

def calc_multi_col_cell_index(cell_index):
    
    multi_col_cell_index = np.zeros(13*nofcols)
    
    for i in range(nofcols):
        
        multi_col_cell_index[i*13:i*13+13] = cell_index+i*65
        
    return multi_col_cell_index.astype(int)

mc_cell_index = calc_multi_col_cell_index(cell_index)

def read_single_connM(fname):
    
    print('reading data from ' + fname + ' file')
    
    data=np.genfromtxt(dirname+fname, delimiter=',', dtype=None)
    
    connM=np.zeros((nofcells,nofcells))
    ctype = ["" for x in range(nofcells)]
    
    for i in range(nofcells):
        ctype[i]=data[i+1,0].decode()
    
    for i in range(1,nofcells+1,1):
        for j in range(1,nofcells+1,1):
            if data[i,j] == b'':
                connM[i-1,j-1] = 0
            else:
                connM[i-1,j-1]=float(data[i,j])
            
    connM=np.transpose(connM)
    
    return connM, ctype

def calcAdjM(M,thr=4):
    
    mydim=M.shape[0]

    AdjM=np.zeros((mydim,mydim))
    
    for i in range(mydim):
        for j in range(mydim):
            if abs(M[i,j]) > thr:
                AdjM[i,j]=1
                
    return AdjM

def read_ConnMs(n=5):

    intra_colM, ctype = read_single_connM(fname1)
    inter_colM, ctype = read_single_connM(fname2)
    
    def create_multi_colM():
    
        multi_colM  = np.zeros((n*nofcells,n*nofcells))
        
        for i in range(n):
            
            left_start  = (i-1)*nofcells
            left_stop   = i*nofcells
            
            centr_start = i*nofcells
            centr_stop  = (i+1)*nofcells
            
            right_start = (i+1)*nofcells
            right_stop  = (i+2)*nofcells
            
            multi_colM[centr_start:centr_stop,centr_start:centr_stop] = intra_colM
            
            if i > 0:
            
                multi_colM[left_start:left_stop,centr_start:centr_stop] = inter_colM/2.0
                
            if i < n-1:
                
                multi_colM[right_start:right_stop,centr_start:centr_stop] = inter_colM/2.0
            
        return multi_colM
    
    multi_colM  = create_multi_colM()
    
    return multi_colM,intra_colM,inter_colM,ctype

def normalize_data(x):
    
    x = x-x[0]
    
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

def read_RecF_data():
    
    # cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])
    
    RF_center_width  = np.array([6,7,6,8,7,6,12,6,6,8,8,11,7])
    RF_surrnd_width  = np.array([41,29,15,33,31,29,7,16,24,27,31,35,24])
    RF_surrnd_weight = np.array([0.012,0.013,0.19,0.046,0.035,0.022,0.000,0.132,0.063,0.040,0.035,0.054,0.046])*5.0
    RF_sign          = np.array([-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1])
    
    RecF_data = np.zeros((13,45))
    
    for i in range(13):
        
        center = bs.Gauss1D(RF_center_width[i],44)
        surrnd = bs.Gauss1D(RF_surrnd_width[i],44)
        
        RecF_data[i]=(center-RF_surrnd_weight[i]*surrnd)*RF_sign[i]
        RecF_data[i]=normalize_data(RecF_data[i])
        
    # hp and lp time constants * 10 ms
    
    IR_hp = np.array([39.1,28.8,00.0,38.1,12.7,31.8,26.0,0.00,0.00,29.6,15.3,24.9,0.00])
    IR_lp = np.array([03.8,05.8,05.4,02.3,04.2,05.4,02.7,03.8,07.7,04.4,01.4,02.4,10.7])
    
    signal=np.zeros(200)
    signal[50:200] =  1.0
    signal = bs.lowpass(signal,5)
    signal = signal / np.max(signal)
    
    ImpR_data = np.zeros((13,200))
    
    for i in range(13):
        
        if IR_hp[i] == 0:
            
            ImpR_data[i] = bs.lowpass(signal,IR_lp[i])
            
        else:
            
            ImpR_data[i] = bs.bandpass(signal,IR_hp[i],IR_lp[i])
            
        # L1 and L2
            
        if i < 2: 
            
            ImpR_data[i] = ImpR_data[i] + 0.4 * signal 
            
        ImpR_data[i] = normalize_data(ImpR_data[i])
        
    # putting it all into a 13 (celltype) x 9 ((space) x 200 (time) array
    
    data = np.zeros((13,9,200))
    
    for i in range(13):
        for j in range(9):
            data[i,j] = RecF_data[i,j*5+2]*ImpR_data[i]
    
    return data

def create_multi_ctype(ctype,n=9):

    multi_ctype = n*ctype
    
    label = ['_'+np.str(x+1) for x in range(n)]
    
    for j in range(n):
        for i in range(nofcells):
            multi_ctype[j*nofcells+i] = ctype[i] + label[j]
            
    return multi_ctype
     
def plot_ConnM():
    
    multi_colM = np.load('Circuits/multi_colM.npy')
    intra_colM = np.load('Circuits/intra_colM.npy')
    inter_colM = np.load('Circuits/inter_colM.npy')
    ctype      = np.load('Circuits/ctype.npy')
    
    mynofcells= multi_colM.shape[0]
    
    n = int(mynofcells/65)
    
    plt.figure(figsize=(20,10))  
    
    # -------------------------------------------------
    
    bs.setmyaxes(0.05,0.55,0.4,0.4)
    
    plt.imshow(intra_colM,vmin=-10,vmax=10,cmap='coolwarm',interpolation='None') 
    plt.xticks(np.arange(65), ctype, rotation=90, fontsize=6)
    plt.yticks(np.arange(65), ctype, rotation=00, fontsize=6)
    plt.title('intra column connectivity',fontsize=14,color='green')
    
    # -------------------------------------------------
    
    bs.setmyaxes(0.05,0.05,0.4,0.4)
    
    plt.imshow(inter_colM,vmin=-10,vmax=10,cmap='coolwarm',interpolation='None') 
    plt.xticks(np.arange(65), ctype, rotation=90, fontsize=6)
    plt.yticks(np.arange(65), ctype, rotation=00, fontsize=6)
    plt.title('inter column connectivity',fontsize=14,color='orange')
    
    # -------------------------------------------------
    
    bs.setmyaxes(0.38,0.05,0.6,0.9)
    
    plt.imshow(multi_colM,vmin=-10,vmax=10,cmap='coolwarm',interpolation='None') 
    plt.axis('off')
    plt.title('overall connectivity',fontsize=14)
    
    # frame it
    
    plt.plot([-0.5, 584.5],[-0.5,-0.5],color='black')
    plt.plot([-0.5, 584.5],[584.5,584.5],color='black')
    plt.plot([-0.5,-0.5],[-0.5, 584.5],color='black')
    plt.plot([584.5,584.5],[-0.5, 584.5],color='black')
    
    # add grid
    
    for i in range(1,n):
            
        plt.plot([i*65,i*65],[0,n*65],color='black',linestyle='dashed')
        plt.plot([0,n*65],[i*65,i*65],color='black',linestyle='dashed')
            
    # add outline
    
    def draw_square(x,y,mycolor):
        
        plt.plot([x,x+63],[y,y],color=mycolor,linewidth=2)
        plt.plot([x,x+63],[y+63,y+63],color=mycolor,linewidth=2)
        plt.plot([x,x],[y,y+63] ,color=mycolor,linewidth=2)
        plt.plot([x+63,x+63],[y,y+63],color=mycolor,linewidth=2)
        
            
    for i in range(n):
        
        # center
        
        x = i*65+1
        y = i*65+1
        
        draw_square(x,y,'green')
        
        #left
        
        x = (i-1)*65+1
        y = i*65+1
        
        draw_square(x,y,'orange')
        
        #right
        
        x = (i+1)*65+1
        y = i*65+1
        
        draw_square(x,y,'orange')
        
    plt.xlim(-0.5,n*65-0.5)
    plt.ylim(n*65-0.5,-0.5)
    
    cbar = plt.colorbar()
    cbar.set_label('inhib      # of synapses      excit', rotation=90, fontsize=12)
    
# ----------------------stimulus generation -----------------------------------

def calc_grating(velo):
    
    maxtime = 1000
    deltat  = 10.0
    nofcols = 5
    
    # wavelength = 30 deg

    movie=np.zeros((180,maxtime))
    image=np.sin(np.arange(1800)/1800.0*2.0*np.pi*6) # 0.1 deg
    image=1.0*(image>0)
    myfilter=bs.Gauss1D(50,200)
    image=np.convolve(image,myfilter)
    image=image[0:1800]
    myvelo=np.zeros(maxtime)
    myvelo[100:maxtime]=velo*10.0/(1000.0/deltat)
    
    for i in range(maxtime):
        
        interim=np.roll(image,int(sum(myvelo[0:i])),axis=0)
        movie[:,i]=bs.rebin(interim,180)
        
    signal = np.zeros((nofcols*65,maxtime))
    
    for i in range(nofcols):
        
        signal[i*65:i*65+8] = movie[5*i+70]
        
    return signal

def calc_edge(velo, polarity = 'bright'):
    
    maxtime = 1000
    deltat  = 10.0
    nofcols = 5

    movie=np.zeros((180,maxtime))
    image=np.zeros(3600)
    image[0:1800]=1.0
    
    if polarity == 'dark':
        image=1.0-image
        
    myfilter=bs.Gauss1D(50,200)
    image=np.convolve(image,myfilter)
    image=image[100:3700]
    
    tstart = int(500-9000.0/velo)
    tstop  = int(500+9000.0/velo)
    
    if tstart < 0:
        tstart= 0
        tstop = maxtime
        image = np.roll(image,150)
    
    myvelo=np.zeros(maxtime)
    myvelo[tstart:tstop]=velo*10.0/(1000.0/deltat)
    
    for i in range(maxtime):
        interim=np.roll(image,int(sum(myvelo[0:i])),axis=0)
        movie[:,i]=bs.rebin(interim[1800:3600],180)
        
    signal = np.zeros((nofcols*65,maxtime))
    
    for i in range(nofcols):
        
        signal[i*65:i*65+8] = movie[5*i+70]
        
    return signal

def calc_bar(velo, polarity = 'bright'):
    
    maxtime = 1000
    deltat  = 10.0
    nofcols = 5

    movie=np.zeros((180,maxtime))
    image=np.zeros(3600)
    image[1800:1850]=1.0
    
    if polarity == 'dark':
        image=1.0-image
        
    myfilter=bs.Gauss1D(50,200)
    image=np.convolve(image,myfilter)
    image=image[100:3700]
    
    tstart = int(500-8750.0/velo)
    tstop  = int(500+8750.0/velo)
    
    if tstart < 0:
        tstart= 0
        tstop = maxtime
        image = np.roll(image,126)
    
    myvelo=np.zeros(maxtime)
    myvelo[tstart:tstop]=velo*10.0/(1000.0/deltat)
    
    for i in range(maxtime):
        interim=np.roll(image,int(sum(myvelo[0:i])),axis=0)
        movie[:,i]=bs.rebin(interim[1800:3600],180)
        
    signal = np.zeros((nofcols*65,maxtime))
    
    for i in range(nofcols):
        
        signal[i*65:i*65+8] = movie[5*i+70]
        
    return signal

def calc_chirp(method='logarithmic',loc_global='global'):
    
    maxtime = 1000
    deltat  = 10.0
    nofcols = 5
    
    myt = np.arange(maxtime)*0.001*deltat
    
    mychirp=chirp(myt,f0=0.1,t1=10,f1=10,method=method)+1.0
    
    signal = np.zeros((nofcols*65,maxtime))
    
    if loc_global == 'global':
    
        for i in range(nofcols):
            
            signal[i*65:i*65+8] = mychirp*0.5
            
    if loc_global == 'local':
            
        signal[4*65:4*65+8] = mychirp*0.5
        
    return signal


        
        
    
    
