# %% [markdown]
# # Connectome Network Visualization
# 
# **Date**: 2026-04-04  
# **Purpose**: Visualize connectome networks using NetworkX
# 
# This notebook creates network visualizations where:
# - **Nodes** = neuron types (cell types)
# - **Node size** = number of neurons of that type
# - **Edge width** = number of synapses between types
# - **Node color** = neurotransmitter type
#   - Red: Excitatory (Acetylcholine, Octopamine, Serotonin, Dopamine)
#   - Blue: Inhibitory (GABA, Glutamate, Histamine)
#   - Gray: Unknown
# 
# Different synapse count thresholds are used to filter weak connections.

# %% [markdown]
# ## Setup

# %%
import sys
sys.path.insert(0, '/Users/lengyuner/Desktop/NIPS2026')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from IPython.display import Image, display

from neuro_framework.connectome.loader import ConnectomeLoader

# %% [markdown]
# ## 1. Optic Lobe Network (925 neurons, 25 cell types)
# 
# Visualization of the complete optic lobe connectome from maleCNS dataset.

# %%
# Display networks with different thresholds
thresholds = [5, 10, 20, 50, 100]

print("Optic Lobe Networks - Different Synapse Thresholds\n")
print("="*70)

for threshold in thresholds:
    img_path = f"network_optic_lobe_threshold_{threshold}.png"
    if Path(img_path).exists():
        print(f"\n### Threshold: ≥{threshold} synapses\n")
        display(Image(filename=img_path, width=800))
    else:
        print(f"Image not found: {img_path}")

# %% [markdown]
# ### Observations - Optic Lobe
# 
# - **25 cell types** including T4/T5 (direction-selective), Mi/Tm (medulla), L1-L5 (lamina)
# - **Network density decreases** with higher thresholds
# - **Hub neurons** (large nodes) include Mi1, Tm3, T4/T5 subtypes
# - **Strong connections** (thick edges) between lamina → medulla → lobula pathway
# - **Color coding** shows mix of excitatory (red) and inhibitory (blue) neurons

# %% [markdown]
# ## 2. T4/T5 Pathway Network (695 neurons, 14 cell types)
# 
# Focused visualization of the motion detection pathway: L1-L5 → Mi/Tm → T4/T5 → LC

# %%
print("T4/T5 Pathway Networks - Different Synapse Thresholds\n")
print("="*70)

for threshold in thresholds:
    img_path = f"network_t4t5_pathway_threshold_{threshold}.png"
    if Path(img_path).exists():
        print(f"\n### Threshold: ≥{threshold} synapses\n")
        display(Image(filename=img_path, width=800))
    else:
        print(f"Image not found: {img_path}")

# %% [markdown]
# ### Observations - T4/T5 Pathway
# 
# - **14 cell types** in the motion detection pathway
# - **Clear hierarchical structure**: Lamina (L1-L5) → Medulla (Mi/Tm) → Lobula (T4/T5) → LC
# - **T4/T5 subtypes** (a,b,c,d) show direction selectivity
# - **LC neurons** (Lobula Columnar) are downstream targets
# - **Feedforward architecture** with some recurrent connections

# %% [markdown]
# ## 3. FAFB Visual System Network (35,345 neurons, 23 cell types)
# 
# Large-scale visualization from FlyWire FAFB v783 dataset, focusing on visual system cell types.

# %%
print("FAFB Visual System Networks - Different Synapse Thresholds\n")
print("="*70)

for threshold in thresholds:
    img_path = f"network_fafb_visual_threshold_{threshold}.png"
    if Path(img_path).exists():
        print(f"\n### Threshold: ≥{threshold} synapses\n")
        display(Image(filename=img_path, width=800))
    else:
        print(f"Image not found: {img_path}")

# %% [markdown]
# ### Observations - FAFB Visual System
# 
# - **23 cell types** from FAFB dataset (35k neurons)
# - **Highly connected network** with many strong connections
# - **Similar pathway structure** to optic lobe: L → Mi/Tm → T4/T5 → LC
# - **More edges** compared to optic lobe due to larger neuron counts
# - **Consistent with known biology** of Drosophila visual system

# %% [markdown]
# ## Network Statistics Summary

# %%
# Load data and compute statistics
datasets = [
    ("Optic Lobe", ConnectomeLoader.from_optic_lobe()),
    ("T4/T5 Pathway", ConnectomeLoader.from_optic_lobe(
        cell_types=['T4a', 'T4b', 'T4c', 'T4d',
                    'T5a', 'T5b', 'T5c', 'T5d',
                    'Mi1', 'Mi4', 'Mi9',
                    'Tm1', 'Tm2', 'Tm3', 'Tm9', 'Tm20',
                    'L1', 'L2', 'L3', 'L4', 'L5',
                    'LC4', 'LC6', 'LC9', 'LC10', 'LC11'],
        min_syn_count=2
    ))
]

print("\n" + "="*70)
print("Network Statistics Summary")
print("="*70 + "\n")

for name, loader in datasets:
    nodes, edges = loader.load()
    
    print(f"{name}:")
    print(f"  Neurons: {len(nodes):,}")
    print(f"  Cell types: {nodes['cell_type'].nunique()}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Avg synapses per edge: {edges['syn_count'].mean():.1f}")
    print(f"  Max synapses: {edges['syn_count'].max():.0f}")
    print()

# %% [markdown]
# ## Key Findings
# 
# ### Network Structure
# 1. **Hierarchical organization**: Clear feedforward structure from photoreceptors → lamina → medulla → lobula
# 2. **Hub neurons**: Mi1, Tm3, T4/T5 act as central nodes with many connections
# 3. **Recurrent connections**: Some feedback loops, especially in higher-order neurons
# 
# ### Synapse Thresholds
# - **Low threshold (5-10)**: Dense networks, many weak connections
# - **Medium threshold (20-50)**: Balanced view, strong pathways visible
# - **High threshold (100+)**: Sparse networks, only strongest connections
# 
# ### Neurotransmitter Distribution
# - **Mixed excitatory/inhibitory**: Both red (exc) and blue (inh) neurons throughout
# - **Glutamate and Histamine**: Classified as inhibitory in this context
# - **Acetylcholine**: Primary excitatory neurotransmitter
# 
# ### Biological Relevance
# - Networks match known anatomy of Drosophila visual system
# - T4/T5 direction selectivity emerges from specific connectivity patterns
# - LC neurons integrate motion signals for behavior

# %% [markdown]
# ## Methods
# 
# ### Network Construction
# ```python
# # Type-to-type aggregation
# 1. Merge edges with neuron types
# 2. Aggregate synapses by (pre_type, post_type)
# 3. Filter by minimum synapse threshold
# 4. Create directed graph with NetworkX
# ```
# 
# ### Visualization Parameters
# - **Layout**: Spring layout (force-directed)
# - **Node size**: Proportional to neuron count (scaled for visibility)
# - **Edge width**: Proportional to synapse count (0.5-5.0 range)
# - **Edge transparency**: 0.3 (to show overlapping edges)
# - **Arrow style**: Directed edges with arrowheads
# 
# ### Color Scheme
# - **Excitatory**: #DC143C (Crimson)
# - **Inhibitory**: #4169E1 (Royal Blue)
# - **Unknown**: #888888 (Gray)

# %% [markdown]
# ## Next Steps
# 
# 1. **Add more datasets**: BANC whole-brain, full FAFB
# 2. **Interactive visualization**: Use plotly or bokeh for zoom/pan
# 3. **Community detection**: Identify functional modules
# 4. **Comparison with models**: Overlay learned vs anatomical weights
# 5. **Temporal dynamics**: Animate network activity during stimulation

# %% [markdown]
# ---
# **Generated**: 2026-04-04  
# **Script**: `04_network_visualization.py`  
# **Output**: 15 network visualizations (3 datasets × 5 thresholds)
