# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 08:43:49 2018

@author: aborst
"""

import numpy as np
import blindschleiche_py3 as bs
from scipy.signal import chirp

nofcells = 65
nofcols  = 5

from training_config import (
    DELTAT_MS,
    IMPULSE_MAXTIME,
    IMPULSE_MAXTIME_MS,
    T_ON,
    T_ON_MS,
    T_TAIL,
    BORST_CTYPE_NPY,
    BORST_MULTI_COL_M_NPY,
    BORST_INTRA_COL_M_NPY,
    BORST_INTER_COL_M_NPY,
)

I_BASELINE = 20.0  # pA photoreceptor current before T_ON
I_BRIGHT = 40.0    # pA photoreceptor current at bright / on-step peak
I_DARK = 0.0       # pA photoreceptor current at full dark-bar coverage
DATA_AMP = 20.0         # pA scale on ImpR target traces (fit cells)

cell_list=np.array(['L1','L2','L3','L4','L5','Mi1','Tm3','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9'])
N_FIT_CELLS = len(cell_list)

ctype      = np.load(BORST_CTYPE_NPY)


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
LEAK_DEPOL_TYPES = ['L1', 'L2', 'L3']  # [] → all -50 mV
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


def leak_depol_indices():
    return tuple(type_index(n) for n in LEAK_DEPOL_TYPES)


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
    tile target, which samples RecF at non-integer column distances (sqrt(3)) --
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


def read_RecF_data_dark():
    """Dark tile spatial×temporal cube: negated bright ``read_RecF_data()``."""
    return -read_RecF_data()


def borst_tile_impulse_data(tile_T=None, amp=DATA_AMP):
    """RecF×ImpR targets for Borst tile training, shape ``(T, nofcells)``."""
    T = int(tile_T or IMPULSE_MAXTIME)
    mydata = read_RecF_data() * amp
    raw = np.zeros((nofcells, T), dtype=np.float64)
    for col in range(nofcols):
        raw[fit_data_slice(col)] = mydata[:, 2 + col, :T]
    return raw.T


def borst_tile_impulse_data_dark(tile_T=None, amp=DATA_AMP):
    """Dark tile targets: inverted bright RecF×ImpR, shape ``(T, nofcells)``."""
    return -borst_tile_impulse_data(tile_T=tile_T, amp=amp)

    
