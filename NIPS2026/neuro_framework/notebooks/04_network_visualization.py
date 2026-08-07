"""
Connectome Network Visualization using NetworkX

This script creates network visualizations of connectome data where:
- Nodes = neuron types (cell types)
- Node size = number of neurons of that type
- Edge width = number of synapses between types
- Node color = neurotransmitter type (red for excitatory, blue for inhibitory)

Author: Generated for NIPS2026 project
Date: 2026-04-04
"""

import sys
sys.path.insert(0, '/Users/lengyuner/Desktop/NIPS2026')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import networkx as nx
from pathlib import Path

from neuro_framework.connectome.loader import ConnectomeLoader


def get_nt_color(nt_type):
    """
    Get color for neurotransmitter type.
    
    Excitatory (red-ish): Acetylcholine, Octopamine, Serotonin, Dopamine
    Inhibitory (blue-ish): GABA, Glutamate, Histamine
    """
    if pd.isna(nt_type) or nt_type == 'unknown':
        return '#888888'  # Gray for unknown
    
    nt_lower = str(nt_type).lower()
    
    # Inhibitory (blue-ish)
    if any(x in nt_lower for x in ['gaba', 'glut', 'histamine']):
        return '#4169E1'  # Royal blue
    
    # Excitatory (red-ish)
    if any(x in nt_lower for x in ['acetylcholine', 'ach', 'octopamine', 'serotonin', 'dopamine']):
        return '#DC143C'  # Crimson
    
    return '#888888'  # Default gray


def build_type_to_type_network(nodes_df, edges_df, min_syn_threshold=10):
    """
    Build a cell-type-to-cell-type network.
    
    Parameters
    ----------
    nodes_df : pd.DataFrame
        Nodes with columns: node_idx, cell_type, nt_type
    edges_df : pd.DataFrame
        Edges with columns: pre_idx, post_idx, syn_count
    min_syn_threshold : int
        Minimum total synapses between types to include edge
    
    Returns
    -------
    G : nx.DiGraph
        NetworkX directed graph
    node_sizes : dict
        Node sizes (neuron counts)
    node_colors : dict
        Node colors (NT type)
    edge_weights : dict
        Edge weights (synapse counts)
    """
    # Merge edges with node info
    edges_with_types = edges_df.copy()
    
    # Merge pre-synaptic info
    edges_with_types = edges_with_types.merge(
        nodes_df[['node_idx', 'cell_type']].rename(columns={'cell_type': 'pre_type'}),
        left_on='pre_idx',
        right_on='node_idx',
        how='left'
    ).drop(columns=['node_idx'])
    
    # Merge post-synaptic info
    edges_with_types = edges_with_types.merge(
        nodes_df[['node_idx', 'cell_type']].rename(columns={'cell_type': 'post_type'}),
        left_on='post_idx',
        right_on='node_idx',
        how='left'
    ).drop(columns=['node_idx'])
    
    # Aggregate by type-to-type
    type_to_type = edges_with_types.groupby(['pre_type', 'post_type']).agg({
        'syn_count': 'sum'
    }).reset_index()
    
    # Filter by threshold
    type_to_type = type_to_type[type_to_type['syn_count'] >= min_syn_threshold]
    
    # Count neurons per type
    type_counts = nodes_df.groupby(['cell_type', 'nt_type']).size().reset_index(name='n_neurons')
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    node_sizes = {}
    node_colors = {}
    for _, row in type_counts.iterrows():
        cell_type = row['cell_type']
        n_neurons = row['n_neurons']
        nt_type = row['nt_type']
        
        G.add_node(cell_type)
        node_sizes[cell_type] = n_neurons
        node_colors[cell_type] = get_nt_color(nt_type)
    
    # Add edges
    edge_weights = {}
    for _, row in type_to_type.iterrows():
        pre = row['pre_type']
        post = row['post_type']
        weight = row['syn_count']
        
        if pre in G.nodes and post in G.nodes:
            G.add_edge(pre, post, weight=weight)
            edge_weights[(pre, post)] = weight
    
    return G, node_sizes, node_colors, edge_weights


def plot_network(G, node_sizes, node_colors, edge_weights, 
                 title="Connectome Network", 
                 figsize=(20, 20),
                 layout='spring',
                 k=2.0,
                 iterations=50):
    """
    Plot network using NetworkX.
    
    Parameters
    ----------
    G : nx.DiGraph
        NetworkX graph
    node_sizes : dict
        Node sizes
    node_colors : dict
        Node colors
    edge_weights : dict
        Edge weights
    title : str
        Plot title
    figsize : tuple
        Figure size
    layout : str
        Layout algorithm: 'spring', 'circular', 'kamada_kawai'
    k : float
        Spring layout parameter (optimal distance between nodes)
    iterations : int
        Number of iterations for spring layout
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=k, iterations=iterations, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Prepare node attributes
    node_list = list(G.nodes())
    sizes = [node_sizes.get(n, 100) for n in node_list]
    colors = [node_colors.get(n, '#888888') for n in node_list]
    
    # Scale node sizes for visualization
    size_scale = 3000 / max(sizes) if max(sizes) > 0 else 1
    sizes_scaled = [s * size_scale for s in sizes]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_size=sizes_scaled,
        node_color=colors,
        alpha=0.8,
        ax=ax
    )
    
    # Draw edges with varying width
    if edge_weights:
        max_weight = max(edge_weights.values())
        min_weight = min(edge_weights.values())
        
        for (u, v), weight in edge_weights.items():
            # Scale edge width
            width = 0.5 + 4.5 * (weight - min_weight) / (max_weight - min_weight + 1e-8)
            
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=0.3,
                edge_color='gray',
                arrows=True,
                arrowsize=10,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DC143C', label='Excitatory (Ach, Oct, 5HT, DA)'),
        Patch(facecolor='#4169E1', label='Inhibitory (GABA, Glut, Hist)'),
        Patch(facecolor='#888888', label='Unknown')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    return fig, ax


def create_network_visualizations(loader, output_dir, dataset_name="Dataset"):
    """
    Create multiple network visualizations with different thresholds.
    
    Parameters
    ----------
    loader : ConnectomeLoader
        Data loader
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name for titles
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n=== Creating network visualizations for {dataset_name} ===\n")
    
    # Load data
    nodes, edges = loader.load()
    print(f"Loaded {len(nodes):,} neurons, {len(edges):,} edges")
    print(f"Cell types: {nodes['cell_type'].nunique()}")
    
    # Different thresholds
    thresholds = [5, 10, 20, 50, 100]
    
    for threshold in thresholds:
        print(f"\nProcessing threshold: {threshold} synapses")
        
        # Build network
        G, node_sizes, node_colors, edge_weights = build_type_to_type_network(
            nodes, edges, min_syn_threshold=threshold
        )
        
        print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if G.number_of_nodes() == 0:
            print(f"  Skipping (no nodes)")
            continue
        
        # Plot with spring layout
        fig, ax = plot_network(
            G, node_sizes, node_colors, edge_weights,
            title=f"{dataset_name} Connectome Network (≥{threshold} synapses)",
            figsize=(20, 20),
            layout='spring',
            k=2.0,
            iterations=100
        )
        
        output_path = output_dir / f"network_{dataset_name.lower().replace(' ', '_')}_threshold_{threshold}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {output_path}")
    
    print(f"\n=== Completed {dataset_name} ===\n")


def main():
    """Main function to create all visualizations."""
    
    # Output directory
    output_dir = Path("/Users/lengyuner/Desktop/NIPS2026/neuro_framework/notebooks")
    
    # 1. Optic Lobe (small, for testing)
    print("\n" + "="*70)
    print("1. Optic Lobe Network")
    print("="*70)
    
    loader_ol = ConnectomeLoader.from_optic_lobe()
    create_network_visualizations(
        loader_ol,
        output_dir,
        dataset_name="Optic Lobe"
    )
    
    # 2. BANC subset (T4/T5 pathway)
    print("\n" + "="*70)
    print("2. BANC T4/T5 Pathway Network")
    print("="*70)
    
    loader_banc = ConnectomeLoader.from_optic_lobe(
        cell_types=['T4a', 'T4b', 'T4c', 'T4d',
                    'T5a', 'T5b', 'T5c', 'T5d',
                    'Mi1', 'Mi4', 'Mi9',
                    'Tm1', 'Tm2', 'Tm3', 'Tm9', 'Tm20',
                    'L1', 'L2', 'L3', 'L4', 'L5',
                    'LC4', 'LC6', 'LC9', 'LC10', 'LC11'],
        min_syn_count=2
    )
    create_network_visualizations(
        loader_banc,
        output_dir,
        dataset_name="T4T5 Pathway"
    )
    
    # 3. FAFB visual system subset
    print("\n" + "="*70)
    print("3. FAFB Visual System Network")
    print("="*70)
    
    try:
        loader_fafb = ConnectomeLoader.from_fafb(
            data_dir="/Users/lengyuner/Desktop/data/flywire/Jun2025",
            super_classes=['optic'],
            cell_types=['L1', 'L2', 'L3', 'L4', 'L5',
                        'Mi1', 'Mi4', 'Mi9',
                        'Tm1', 'Tm2', 'Tm3', 'Tm9', 'Tm20',
                        'T4a', 'T4b', 'T4c', 'T4d',
                        'T5a', 'T5b', 'T5c', 'T5d',
                        'LC4', 'LC6', 'LC9', 'LC10', 'LC11',
                        'C2', 'C3'],
            min_syn_count=5
        )
        create_network_visualizations(
            loader_fafb,
            output_dir,
            dataset_name="FAFB Visual"
        )
    except Exception as e:
        print(f"Skipping FAFB: {e}")
    
    print("\n" + "="*70)
    print("All visualizations completed!")
    print("="*70)


if __name__ == "__main__":
    main()
