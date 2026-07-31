# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:53:25 2023

@author: aborst
"""
import numpy as np
import matplotlib.pyplot as plt
import Medulla_Library as ml
import time

import torch
from torch import nn
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################################################
# Medulla Library contains:
# ml.read_ConnMs()
# ml.read_RecF_data(): RecF_data (13,45), ImpR_data (13,45)
# plot_ConnM(): Big ConnM + intra + inter M
# stimulus generation -> signal
#################################################################

plt.rcParams['axes.facecolor'] = '#EEEEEE'
plt.rcParams['figure.facecolor'] = 'white'

nofcells  = 65
nofcols   = 5
maxtime   = 200

# important model params

deltat    = 10.0  # in msec
g_leak    = 1.0   # in nS
E_exc     = +10.0 # in mV
E_inh     = -70.0 # in mV
capac     = +40.0 # in pF, results in 50ms membrane time-constant for g_leak = 1.0 nS
trld      = -50.0 # in mV: below trld, no signal is transmitted
cdt       = capac/deltat

Ca_tau    = 50.0  # in msec

E_leak = torch.zeros(nofcells * nofcols, dtype=torch.float64).to(device) - 50.0

for i in range(nofcols):
    
    # Lamina cells L1-L3 get E_leak of - 20.0 mV
    
    E_leak[nofcells*i+8:nofcells*i+11]  = -20.0

exc_synweight = 0.001
inh_synweight = 0.001

# ----------- H-Current ----------------------------------------

E_Ih          = +50.0  # in mV
Ih_midv       = -50.0
Ih_slope      = -0.25
tau_midv      = -50.0
Ih_gmax       = +50.0 

Ih_gain       = 1.0   # if set to 0, it will block Ih

signal_amp    = 40.0 # amplitude of current injection in photoreceptors, in pA.
data_amp      = 20.0  # amplitude of impulse response of all cells

# parameter and cost function definition

nofparams = 2 * nofcells + 8 # 8 params for Ih (5 x gmax + 3)

low_gain = 0.1
high_gain = 100.0

# -------------------------------------------------------------------------
# -------------- reading cell data and connectivity matrices --------------
# -------------------------------------------------------------------------

def init_network():
    
    multi_colM    = np.load('Circuits/multi_colM.npy')
    ctype         = np.load('Circuits/ctype.npy')
    mc_cell_index = np.load('Circuits/mc_cell_index.npy')
    
    multi_colM[130+11,0:130] = 0
    multi_colM[130+11,195:325] = 0
    
    # chemical synpases

    M_exc = exc_synweight * multi_colM * (multi_colM > 0)
    M_inh = inh_synweight * multi_colM * (multi_colM < 0) * (-1)
    
    M_exc = torch.tensor(M_exc,dtype=torch.float64).to(device)
    M_inh = torch.tensor(M_inh,dtype=torch.float64).to(device)
    
    signal = torch.zeros((200,325), dtype=torch.float64).to(device)
    signal[50:200,130:138,] = signal_amp

    mydata = ml.read_RecF_data()*data_amp
    mydata = torch.tensor(mydata, dtype=torch.float64)
    data   = torch.zeros((65,maxtime), dtype=torch.float64).to(device)
    
    for i in range(5):
        
        data[i*13:13+i*13] = mydata[:,2+i]
        
    data = torch.transpose(data,0,1)
        
    power = torch.sum((data[50:200])**2)
    
    return M_exc, M_inh, ctype, mc_cell_index, data, power, signal

M_exc, M_inh, ctype, mc_cell_index, data, power, signal = init_network()
        
# ------- network calculations  -----------------------------------------------

def calc_multi_col_params(param):
    
    # turns params for one column into multi-column params
    
    multi_col_param = torch.concatenate((param,param,param,param,param))
        
    return multi_col_param

def rectsyn(x,thrld):
    
    result=x-thrld
    result=result*(result>0)
    
    return result

def update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,signal):

    Ih_ss   = 1.0/(1.0+torch.exp((Ih_midv-Vm)*Ih_slope))
    tau     = 1.5/(torch.exp(-0.1*(Vm-tau_midv))+torch.exp(+0.1*(Vm-tau_midv)))*1000.0 + 100.0
    u       = deltat/tau*(Ih_ss-u)+u
    g_Ih    = u * Ih_gmax * Ih_gain
    
    g_exc   = torch.mv(M_exc,(rectsyn(Vm,trld)*out_gain))*inp_gain
    g_inh   = torch.mv(M_inh,(rectsyn(Vm,trld)*out_gain))*inp_gain
    
    Vm = (g_exc*E_exc + g_inh*E_inh + g_leak*E_leak + E_Ih * g_Ih + cdt*Vm + signal)
    Vm = Vm / (g_exc + g_inh + g_Ih + g_leak + cdt)
    
    return Vm, u

#@torch.compile
def calc_cost(z,data):
    
    cost = 0
    
    inp_gain = calc_multi_col_params(z[0:65]).to(device)
    out_gain = calc_multi_col_params(z[65:130]).to(device)
    
    interim = torch.zeros(65, dtype=torch.float64).to(device)
    interim[8:13] = z[130:135]
    Ih_gmax = calc_multi_col_params(interim).to(device)
    
    Ih_midv  = z[135]
    Ih_slope = z[136]
    tau_midv = z[137]
    
    u  = torch.zeros(325, dtype=torch.float64).to(device)

    Vm = E_leak
    
    model = 0
    
    for t in range(1,50): 
        
        Vm, u = update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,signal[t-1])

    Vm_t49 = 1.0*Vm[mc_cell_index] # take this Vm as reference 0
        
    for t in range(50,200): 
        
        Vm, u = update_Vm(Vm,u,inp_gain,out_gain,Ih_gmax,Ih_midv,Ih_slope,tau_midv,signal[t-1])
        
        model = deltat/Ca_tau * (Vm[mc_cell_index] - Vm_t49 - model) + model # low-pass filter
        
        cost = cost + torch.sum((model-data[t])**2)
        
    cost = cost/power*100.0
        
    return cost
    
def calc_z_bounds():
    
    z_bounds = torch.zeros((nofparams,2), dtype=torch.float64)
    
    # input and output gain ------------------
    
    for i in range(2*nofcells): 
        
        z_bounds[i] = torch.tensor([low_gain, high_gain], dtype=torch.float64)
        
    # Ih_gmax
    
    for i in range(5):
    
        z_bounds[130+i] = torch.tensor([0,100], dtype=torch.float64)
    
    # Ih_midv
    
    z_bounds[135] = torch.tensor([-70,-30], dtype=torch.float64)
    
    # Ih_slope
    
    z_bounds[136] = torch.tensor([-0.40,-0.20], dtype=torch.float64)
    
    # Ih_tau
    
    z_bounds[137] = torch.tensor([-70,-40], dtype=torch.float64)
    
    return z_bounds

z_bounds = calc_z_bounds()
    
def guess_initial_params():
    
    z = np.zeros(nofparams)
        
    z[0:65]    = + 0.5    + (np.random.rand(65)-0.5)*0.2
    z[65:130]  = + 0.5    + (np.random.rand(65)-0.5)*0.2
    z[130:135] = Ih_gmax  + (np.random.rand(5)-0.5)*10.0    # L1,L2,L5
    z[132:134] = 0.0                                        # L3,L4
    z[135]     = Ih_midv  + (np.random.rand()-0.5)*5.0
    z[136]     = Ih_slope + (np.random.rand()-0.5)*0.02
    z[137]     = tau_midv + (np.random.rand()-0.5)*5.0
    
    z = torch.tensor(z, dtype=torch.float64).to(device)
    
    return z

def gradient_network(data, z, lr=0.0001, cost_fn=None, n_steps=100, device="cpu", z_bounds=None):
    
    a = time.time()

    z = nn.Parameter(z.clone().to(device))
    data = data.to(device)
    
    optimizer = torch.optim.Adam([z], lr=lr)

    # Calculate initial cost and move it to the specified device
    
    cost = cost_fn(z, data).item()
    best_cost = cost
    best_z = z.clone().detach()
    
    initial_cost = 1.0*cost

    progress_bar = tqdm(range(n_steps), desc=f'Cost: {cost:.4f}')

    for i in progress_bar:
        
        optimizer.zero_grad()
        
        cost = cost_fn(z, data)  
        
        if cost.item() < best_cost:
            
            best_cost = cost.item()
            best_z = z.clone().detach()
        
        cost.backward()
        optimizer.step()

        with torch.no_grad():
            
            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

        progress_bar.set_description(f'Cost: {cost.item():.4f}')

    cost = cost_fn(z, data)  
    
    if cost.item() < best_cost:
        
        best_cost = cost.item()
        best_z = z.clone().detach()

    print()
    print('Initl cost =', format(initial_cost,'.4f'))
    print('Final cost =', format(cost.item(),'.4f'))
    print('Best  cost =', format(best_cost,'.4f'))
    
    b = time.time()
    
    print('time needed  =',format(b-a,'.2f'),' sec')
    print()

    return best_z

def do_many_runs(nofruns,nofsteps,fname):
    
    dirname = 'FiveCol_Parameter/'
    
    all_params = np.zeros((nofruns,nofparams))
    
    for i in range(nofruns):
        
        print()
        print('round',i)
        print()
        
        z = guess_initial_params()

        z_fit = gradient_network(
            data,
            z,
            lr=0.1,
            n_steps=nofsteps,
            device=device,
            cost_fn=calc_cost,
            z_bounds = z_bounds
        )
        
        z_fit = gradient_network(
            data,
            z_fit,
            lr=0.01,
            n_steps=nofsteps,
            device=device,
            cost_fn=calc_cost,
            z_bounds = z_bounds
        )
        
        z_fit = gradient_network(
            data,
            z_fit,
            lr=0.001,
            n_steps=nofsteps,
            device=device,
            cost_fn=calc_cost,
            z_bounds = z_bounds
        )
        
        all_params[i] = z_fit.detach().numpy()
        
        np.save(dirname+fname,all_params)
        
def refine_many_runs(all_params,nofsteps,lr = 0.01):
    
    dirname = 'FiveCol_Parameter/'
    fname   = 'refined_paramsets.npy'
    
    nofruns = all_params.shape[0]
    
    ref_params = np.zeros((nofruns,nofparams))
    
    for i in range(nofruns):
        
        print()
        print('round',i)
        print()
        
        z = all_params[i]
        z = torch.tensor(z).double().requires_grad_()

        z_fit = gradient_network(
            data,
            z,
            lr=lr,
            n_steps=nofsteps,
            device=device,
            cost_fn=calc_cost,
            z_bounds = z_bounds
        )
        
        ref_params[i] = z_fit.detach().numpy()
        
        np.save(dirname+fname,ref_params)
        
def save_numpy_parameters(z_fit,fname):
    
    dirname = 'FiveCol_Parameter/'
    z = z_fit.detach().numpy()
    np.save(dirname+fname,z)
    
    
if __name__ == "__main__":
    
    dirname = 'FiveCol_Parameter/with_Ih/'

    fname = '4Ih_paramset_L4isol_NoGaps.npy'
    z = np.load(dirname+fname)
    z = torch.tensor(z, dtype=torch.float64).to(device)
    
    #z = guess_initial_params()
    
    z_fit = gradient_network(
        data,
        z,
        lr=0.001,
        n_steps=10,
        device=device,
        cost_fn=calc_cost,
        z_bounds = z_bounds
    )
    # print(f"final cost: {calc_cost(z_fit,data)} , initial cost {calc_cost(z,data)}")



    



