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

# Single source of truth for simulation / training horizon (10 ms per step).
IMPULSE_MAXTIME = 200
T_ON = 50
SIGNAL_BASELINE = 20.0  # pA photoreceptor current before T_ON
SIGNAL_BRIGHT = 40.0    # pA photoreceptor current at bright / on-step peak
SIGNAL_DARK = 0.0       # pA photoreceptor current at full dark-bar coverage
DATA_AMP = 20.0         # pA scale on ImpR target traces (fit cells)

cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])
N_FIT_CELLS = len(cell_list)

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
        
        multi_col_cell_index[i * N_FIT_CELLS:(i + 1) * N_FIT_CELLS] = cell_index + column_start(i)
        
    return multi_col_cell_index.astype(int)


# --- 5-column Borst layout (single source of truth) -------------------------
CENTER_COL = 2
N_PHOTORECEPTORS = 8
LAMINA_SLICE = slice(8, 13)           # L1-L5 within the 65-type vocabulary
LAMINA_DEPOL_TYPES = (8, 9, 10)       # L1-L3 resting depolarisation
# Center-column fit cells whose lateral presynaptic input is zeroed in multi_colM.
ISOLATED_CENTER_FIT_TYPES = ('Tm4',)


def n_state_units(n_cols=None):
    cols = nofcols if n_cols is None else n_cols
    return cols * nofcells


def column_slice(col: int) -> slice:
    start = col * nofcells
    return slice(start, start + nofcells)


def column_start(col: int) -> int:
    return col * nofcells


mc_cell_index = calc_multi_col_cell_index(cell_index)


def unit_index(col: int, type_idx) -> int:
    return col * nofcells + int(type_idx)


def center_unit_index(type_idx) -> int:
    return unit_index(CENTER_COL, type_idx)


def type_index(name: str) -> int:
    matches = np.where(ctype == name)[0]
    if len(matches) != 1:
        raise KeyError(f"cell type {name!r} not found uniquely in ctype ({len(matches)} matches)")
    return int(matches[0])


def fit_list_index(name: str) -> int:
    matches = np.where(cell_list == name)[0]
    if len(matches) != 1:
        raise KeyError(f"fit cell {name!r} not in cell_list")
    return int(matches[0])


def fit_type_index(name: str) -> int:
    return int(cell_index[fit_list_index(name)])


def photoreceptor_slice(col: int = CENTER_COL) -> slice:
    start = col * nofcells
    return slice(start, start + N_PHOTORECEPTORS)


def fit_data_slice(col: int) -> slice:
    start = col * N_FIT_CELLS
    return slice(start, (col + 1) * N_FIT_CELLS)


def apply_borst_connectivity_patches(multi_colM: np.ndarray) -> np.ndarray:
    """Zero lateral presynaptic input to configured center-column cell types."""
    for name in ISOLATED_CENTER_FIT_TYPES:
        post = center_unit_index(fit_type_index(name))
        for col in range(nofcols):
            if col == CENTER_COL:
                continue
            multi_colM[post, column_slice(col)] = 0
    return multi_colM


def legacy_conductance_z_slices():
    """Historical 138-parameter conductance z layout (pre-schema Python path)."""
    ih_start = 2 * nofcells
    n_lamina_ih = LAMINA_SLICE.stop - LAMINA_SLICE.start
    return {
        'inp_gain': slice(0, nofcells),
        'out_gain': slice(nofcells, 2 * nofcells),
        'Ih_gmax': slice(ih_start, ih_start + n_lamina_ih),
        'Ih_midv': ih_start + n_lamina_ih,
        'Ih_slope': ih_start + n_lamina_ih + 1,
        'tau_midv': ih_start + n_lamina_ih + 2,
        'n_params': 2 * nofcells + n_lamina_ih + 3,
        'n_selp_correlation': ih_start + n_lamina_ih,
    }


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
            
            left_start  = column_start(i - 1)
            left_stop   = column_start(i)
            
            centr_start = column_start(i)
            centr_stop  = column_start(i + 1)
            
            right_start = column_start(i + 1)
            right_stop  = column_start(i + 2)
            
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

def read_RecF_ImpR():
    """Return (RecF_data (13,45), ImpR_data (13,IMPULSE_MAXTIME)) for the 13 fit cell types.

    Split out of read_RecF_data so callers that need the continuous spatial RF
    (RecF_data) or the temporal kernel (ImpR_data) on their own -- e.g. the hex
    radial target, which samples RecF at non-integer column distances (sqrt(3)) --
    use the EXACT same construction the 5-column model uses (single source).
    """

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
    
    signal = np.zeros(IMPULSE_MAXTIME)
    signal[T_ON:IMPULSE_MAXTIME] = 1.0
    signal = bs.lowpass(signal, 5)
    signal = signal / np.max(signal)

    ImpR_data = np.zeros((13, IMPULSE_MAXTIME))
    
    for i in range(13):
        
        if IR_hp[i] == 0:
            
            ImpR_data[i] = bs.lowpass(signal,IR_lp[i])
            
        else:
            
            ImpR_data[i] = bs.bandpass(signal,IR_hp[i],IR_lp[i])
            
        # L1 and L2
            
        if i < 2: 
            
            ImpR_data[i] = ImpR_data[i] + 0.4 * signal 
            
        ImpR_data[i] = normalize_data(ImpR_data[i])

    return RecF_data, ImpR_data


def read_RecF_data():
    # putting it all into a 13 (celltype) x 9 (space) x IMPULSE_MAXTIME (time) array.
    # space index j maps to RF sample 5j+2 (j=4 -> sample 22 = RF centre, r=0);
    # so column distance r maps to continuous RF sample 22 + 5r.

    RecF_data, ImpR_data = read_RecF_ImpR()

    data = np.zeros((13, 9, IMPULSE_MAXTIME))

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
    
    n_units = n_state_units(n)
    
    for i in range(1, n):
            
        x = column_start(i)
        plt.plot([x, x], [0, n_units], color='black', linestyle='dashed')
        plt.plot([0, n_units], [x, x], color='black', linestyle='dashed')
            
    # add outline
    
    def draw_square(x,y,mycolor):
        
        plt.plot([x,x+63],[y,y],color=mycolor,linewidth=2)
        plt.plot([x,x+63],[y+63,y+63],color=mycolor,linewidth=2)
        plt.plot([x,x],[y,y+63] ,color=mycolor,linewidth=2)
        plt.plot([x+63,x+63],[y,y+63],color=mycolor,linewidth=2)
        
            
    for i in range(n):
        
        # center
        
        x = column_start(i) + 1
        y = column_start(i) + 1
        
        draw_square(x, y, 'green')
        
        #left
        
        x = column_start(i - 1) + 1
        y = column_start(i) + 1
        
        draw_square(x, y, 'orange')
        
        #right
        
        x = column_start(i + 1) + 1
        y = column_start(i) + 1
        
        draw_square(x, y, 'orange')
        
    plt.xlim(-0.5, n_units - 0.5)
    plt.ylim(n_units - 0.5, -0.5)
    
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
        
    signal = np.zeros((n_state_units(nofcols), maxtime))
    
    for i in range(nofcols):
        
        signal[photoreceptor_slice(i)] = movie[5 * i + 70]
        
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
        
    signal = np.zeros((n_state_units(nofcols), maxtime))
    
    for i in range(nofcols):
        
        signal[photoreceptor_slice(i)] = movie[5 * i + 70]
        
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
        
    signal = np.zeros((n_state_units(nofcols), maxtime))
    
    for i in range(nofcols):
        
        signal[photoreceptor_slice(i)] = movie[5 * i + 70]
        
    return signal

def calc_chirp(method='logarithmic',loc_global='global'):
    
    maxtime = 1000
    deltat  = 10.0
    nofcols = 5
    
    myt = np.arange(maxtime)*0.001*deltat
    
    mychirp=chirp(myt,f0=0.1,t1=10,f1=10,method=method)+1.0
    
    signal = np.zeros((n_state_units(nofcols), maxtime))
    
    if loc_global == 'global':
    
        for i in range(nofcols):
            
            signal[photoreceptor_slice(i)] = mychirp * 0.5
            
    if loc_global == 'local':
            
        signal[photoreceptor_slice(4)] = mychirp * 0.5
        
    return signal


        
        
    
    
