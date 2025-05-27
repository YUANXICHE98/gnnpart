import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import torch
from PIL import Image, ImageDraw, ImageFont
import json
import math
from torch_geometric.utils import to_networkx
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import argparse
import datetime


# utils/visualization.py file, containing multiple visualization functions:
# create_paper_visualization - Create paper-level graph sensitivity analysis visualization (Chinese version)
# calculate_node_sensitivity - Calculate node sensitivity based on PageRank algorithm
# visualize_group_comparison - Visualization comparing original and extended graph (Chinese version)
# visualize_group_comparison_english - Graph comparison visualization (English version)
# create_paper_visualization_english - Paper graph structure visualization (English version)
# visualize_sensitive_edges - Visualization focusing on edge sensitivity
# test_graph_extension.py file, containing functions for testing graph extension:
# visualize_group_extension - Generate visualization for group graph extension
# Other functions related to dataset processing and visualization
# Methods related to graph extension in GroupDataset class:
# extend_group_graph - Generic graph extension method
# extend_abelian_graph - Abelian group graph extension
# extend_non_abelian_graph - Non-Abelian group graph extension
# extend_identity_graph - Identity group graph extension



def create_paper_visualization(predictions=None, targets=None, features=None, edge_attrs=None, group_types=None, save_path=None, title="Graph Structure Sensitivity Analysis"):
    """Create professional visualization for papers, showing comparison before and after sensitivity calculation
    
    Args:
        predictions: Array of predictions
        targets: Array of targets
        features: Array of features
        edge_attrs: Array of edge features
        group_types: List of group types
        save_path: Save path
        title: Title of the graph
    """
    plt.figure(figsize=(16, 8))
    plt.suptitle(title, fontsize=16)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    num_nodes = features.shape[0] if features is not None else len(predictions)
    for i in range(num_nodes):
        G.add_node(i)
    
    # Calculate node positions
    pos = nx.spring_layout(G, seed=42)
    
    # 1. Original graph structure
    plt.subplot(1, 2, 1)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        width=2,
        edge_color='grey',
        alpha=0.6,
        arrowsize=15
    )
    
    # Draw nodes - using predictions as colors
    node_colors = predictions if predictions is not None else np.zeros(num_nodes)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=500,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        vmin=0.0,
        vmax=1.0
    )
    
    # Add color bar
    plt.colorbar(nodes, label='Prediction Value')
    
    plt.title("Original Graph Structure")
    plt.axis('off')
    
    # 2. Sensitivity heatmap
    plt.subplot(1, 2, 2)
    
    # Calculate edge sensitivity
    if edge_attrs is not None:
        edge_sensitivities = np.linalg.norm(edge_attrs, axis=1)
        edge_sensitivities = (edge_sensitivities - edge_sensitivities.min()) / (edge_sensitivities.max() - edge_sensitivities.min() + 1e-6)
    else:
        edge_sensitivities = np.zeros(G.number_of_edges())
    
    # Draw edges - using sensitivity as color
    edges = nx.draw_networkx_edges(
        G, pos, 
        width=2,
        edge_color=edge_sensitivities,
        edge_cmap=plt.cm.YlOrRd,
        edge_vmin=0.0,
        edge_vmax=1.0,
        arrowsize=15
    )
    
    # Draw nodes - using features as color
    if features is not None:
        node_features = np.mean(features, axis=1)
        node_features = (node_features - node_features.min()) / (node_features.max() - node_features.min() + 1e-6)
    else:
        node_features = np.zeros(num_nodes)
    
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=500,
        node_color=node_features,
        cmap=plt.cm.YlOrRd,
        vmin=0.0,
        vmax=1.0
    )
    
    # Add color bar
    plt.colorbar(nodes, label='Feature Value')
    
    plt.title("Sensitivity Analysis Heatmap")
    plt.axis('off')
    
    # Add group type information
    if group_types is not None:
        # Convert each element in the list to string
        group_types = [str(gt) for gt in group_types]
        unique_groups = set(group_types)
        group_info = f"Group Types: {', '.join(unique_groups)}"
        plt.figtext(0.5, 0.02, group_info, ha='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save image
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Paper visualization saved to: {save_path}")
    
    plt.close()

def calculate_node_sensitivity(G, edge_sensitivities, propagation_factor=0.85, num_iterations=10):
    """Calculate node sensitivity, based on a PageRank-like propagation algorithm"""
    
    # Initialize node sensitivity
    node_sensitivity = {node: 0.0 for node in G.nodes()}
    
    # Propagate edge sensitivity to related nodes
    for (u, v), sensitivity in edge_sensitivities.items():
        node_sensitivity[u] += sensitivity / 2.0
        node_sensitivity[v] += sensitivity / 2.0
    
    # Normalize initial sensitivity
    total_sensitivity = sum(node_sensitivity.values())
    if total_sensitivity > 0:
        for node in node_sensitivity:
            node_sensitivity[node] /= total_sensitivity
    
    # Use PageRank-like propagation algorithm to update sensitivity iteratively
    for _ in range(num_iterations):
        new_sensitivity = {node: 0.0 for node in G.nodes()}
        
        # Calculate propagation through edges
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                propagated_value = node_sensitivity[node] * propagation_factor / len(neighbors)
                for neighbor in neighbors:
                    new_sensitivity[neighbor] += propagated_value
        
        # Retain a portion of initial sensitivity
        for node in G.nodes():
            new_sensitivity[node] += (1 - propagation_factor) * node_sensitivity[node]
        
        # Update sensitivity values
        node_sensitivity = new_sensitivity
    
    return node_sensitivity

def visualize_group_comparison(original_graph, extended_graph, output_path, group_type="abelian"):
    """
    Create comparative visualization showing the difference between original and extended graphs
    
    Parameters:
    - original_graph: Original PyG graph object
    - extended_graph: Extended PyG graph object
    - output_path: Save path for the output image
    - group_type: Group type name, used for title
    """
    plt.figure(figsize=(15, 12))
    
    # Extract data
    orig_edge_index = original_graph.edge_index.cpu().numpy()
    ext_edge_index = extended_graph.edge_index.cpu().numpy()
    
    # Create NetworkX graphs
    G_orig = nx.Graph()
    G_ext = nx.Graph()
    
    orig_nodes = max(np.max(orig_edge_index[0]), np.max(orig_edge_index[1])) + 1
    ext_nodes = max(np.max(ext_edge_index[0]), np.max(ext_edge_index[1])) + 1
    
    G_orig.add_nodes_from(range(orig_nodes))
    G_ext.add_nodes_from(range(ext_nodes))
    
    # Extract edge sensitivities
    orig_sensitivities = {}
    ext_sensitivities = {}
    
    if hasattr(original_graph, 'edge_attr') and original_graph.edge_attr is not None:
        orig_edge_attr = original_graph.edge_attr.cpu().numpy()
        if orig_edge_attr.shape[1] > 0:
            for i in range(len(orig_edge_index[0])):
                src, dst = orig_edge_index[0, i], orig_edge_index[1, i]
                sensitivity = float(orig_edge_attr[i, 0]) if i < len(orig_edge_attr) else 1.0
                G_orig.add_edge(src, dst, sensitivity=sensitivity)
                orig_sensitivities[(src, dst)] = sensitivity
        else:
            for i in range(len(orig_edge_index[0])):
                src, dst = orig_edge_index[0, i], orig_edge_index[1, i]
                G_orig.add_edge(src, dst, sensitivity=1.0)
                orig_sensitivities[(src, dst)] = 1.0
    else:
        for i in range(len(orig_edge_index[0])):
            src, dst = orig_edge_index[0, i], orig_edge_index[1, i]
            G_orig.add_edge(src, dst, sensitivity=1.0)
            orig_sensitivities[(src, dst)] = 1.0
    
    if hasattr(extended_graph, 'edge_attr') and extended_graph.edge_attr is not None:
        ext_edge_attr = extended_graph.edge_attr.cpu().numpy()
        if ext_edge_attr.shape[1] > 0:
            for i in range(len(ext_edge_index[0])):
                src, dst = ext_edge_index[0, i], ext_edge_index[1, i]
                sensitivity = float(ext_edge_attr[i, 0]) if i < len(ext_edge_attr) else 1.0
                G_ext.add_edge(src, dst, sensitivity=sensitivity)
                ext_sensitivities[(src, dst)] = sensitivity
        else:
            for i in range(len(ext_edge_index[0])):
                src, dst = ext_edge_index[0, i], ext_edge_index[1, i]
                G_ext.add_edge(src, dst, sensitivity=1.0)
                ext_sensitivities[(src, dst)] = 1.0
    else:
        for i in range(len(ext_edge_index[0])):
            src, dst = ext_edge_index[0, i], ext_edge_index[1, i]
            G_ext.add_edge(src, dst, sensitivity=1.0)
            ext_sensitivities[(src, dst)] = 1.0
    
    # Calculate node sensitivity
    orig_node_sens = calculate_node_sensitivity(G_orig, orig_sensitivities)
    ext_node_sens = calculate_node_sensitivity(G_ext, ext_sensitivities)
    
    # Draw original graph
    plt.subplot(2, 2, 1)
    pos_orig = nx.spring_layout(G_orig, seed=42)
    
    orig_node_colors = [orig_node_sens[node] for node in G_orig.nodes()]
    orig_node_sizes = [300 * (0.5 + orig_node_sens[node] / max(orig_node_sens.values(), default=1)) 
                      for node in G_orig.nodes()]
    
    nx.draw_networkx_nodes(G_orig, pos_orig, node_color=orig_node_colors, 
                          node_size=orig_node_sizes, cmap=plt.cm.YlOrRd, alpha=0.8)
    
    orig_edges = list(G_orig.edges(data=True))
    orig_edge_colors = [data['sensitivity'] for _, _, data in orig_edges]
    orig_edge_widths = [1.0 + 2.0 * data['sensitivity'] / max([d['sensitivity'] for _, _, d in orig_edges], default=1) 
                       for _, _, data in orig_edges]
    
    nx.draw_networkx_edges(G_orig, pos_orig, width=orig_edge_widths, 
                          edge_color=orig_edge_colors, edge_cmap=plt.cm.Blues, alpha=0.7)
    
    # Show only a few node labels to avoid crowding
    if len(G_orig.nodes()) > 20:
        labels = {node: str(node) for i, node in enumerate(G_orig.nodes()) if i % 5 == 0}
    else:
        labels = {node: str(node) for node in G_orig.nodes()}
    
    nx.draw_networkx_labels(G_orig, pos_orig, labels=labels, font_size=8, font_color='black')
    
    plt.title(f'Original {group_type.capitalize()} Graph', fontsize=14)
    plt.axis('off')
    
    # Draw extended graph
    plt.subplot(2, 2, 2)
    pos_ext = nx.spring_layout(G_ext, seed=42)
    
    ext_node_colors = [ext_node_sens[node] for node in G_ext.nodes()]
    ext_node_sizes = [300 * (0.5 + ext_node_sens[node] / max(ext_node_sens.values(), default=1)) 
                     for node in G_ext.nodes()]
    
    nx.draw_networkx_nodes(G_ext, pos_ext, node_color=ext_node_colors, 
                          node_size=ext_node_sizes, cmap=plt.cm.YlOrRd, alpha=0.8)
    
    ext_edges = list(G_ext.edges(data=True))
    ext_edge_colors = [data['sensitivity'] for _, _, data in ext_edges]
    ext_edge_widths = [1.0 + 2.0 * data['sensitivity'] / max([d['sensitivity'] for _, _, d in ext_edges], default=1) 
                      for _, _, data in ext_edges]
    
    nx.draw_networkx_edges(G_ext, pos_ext, width=ext_edge_widths, 
                          edge_color=ext_edge_colors, edge_cmap=plt.cm.Blues, alpha=0.7)
    
    # Show only a few node labels
    if len(G_ext.nodes()) > 20:
        labels = {node: str(node) for i, node in enumerate(G_ext.nodes()) if i % 5 == 0}
    else:
        labels = {node: str(node) for node in G_ext.nodes()}
    
    nx.draw_networkx_labels(G_ext, pos_ext, labels=labels, font_size=8, font_color='black')
    
    plt.title(f'Extended {group_type.capitalize()} Graph', fontsize=14)
    plt.axis('off')
    
    # Draw sensitivity distribution - original graph
    plt.subplot(2, 2, 3)
    orig_sens_values = list(orig_sensitivities.values())
    plt.hist(orig_sens_values, bins=20, alpha=0.7, color='blue')
    plt.title('Original Graph Sensitivity Distribution', fontsize=14)
    plt.xlabel('Sensitivity')
    plt.ylabel('Frequency')
    
    # Draw sensitivity distribution - extended graph
    plt.subplot(2, 2, 4)
    ext_sens_values = list(ext_sensitivities.values())
    plt.hist(ext_sens_values, bins=20, alpha=0.7, color='green')
    plt.title('Extended Graph Sensitivity Distribution', fontsize=14)
    plt.xlabel('Sensitivity')
    plt.ylabel('Frequency')
    
    # Add statistics
    plt.figtext(0.5, 0.01, 
                f"Original graph: node count {G_orig.number_of_nodes()}, edge count {G_orig.number_of_edges()}\n" 
                f"Extended graph: node count {G_ext.number_of_nodes()}, edge count {G_ext.number_of_edges()}\n"
                f"Added: node count {G_ext.number_of_nodes() - G_orig.number_of_nodes()}, " 
                f"edge count {G_ext.number_of_edges() - G_orig.number_of_edges()}", 
                ha='center', fontsize=12)
    
    plt.suptitle(f'{group_type.capitalize()} Group Graph Extension Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Group graph comparison visualization saved to: {output_path}")
    return output_path

def calculate_node_sensitivity(graph, num_iterations=10, damping=0.85):
    """
    Calculate node sensitivity based on edge sensitivities using PageRank-like algorithm
    
    Parameters:
    - graph: PyG graph object
    - num_iterations: Number of iterations for propagation
    - damping: Damping factor for propagation
    
    Returns:
    - Dictionary of node sensitivities
    """
    # Convert to NetworkX for easier handling
    nx_graph = to_networkx(graph, to_undirected=True)
    
    # Get edge sensitivities from graph
    edge_sensitivities = {}
    if hasattr(graph, 'edge_sensitivities') and graph.edge_sensitivities is not None:
        edge_sensitivities = graph.edge_sensitivities
    elif hasattr(graph, 'sensitive_edge_mask') and graph.sensitive_edge_mask is not None:
        # Use sensitive edge mask for edge sensitivities
        edge_mask = graph.sensitive_edge_mask.numpy()
        edge_index = graph.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_sensitivities[f"{src},{dst}"] = float(edge_mask[i])
    
    # Initialize node sensitivities
    node_sensitivities = {node: 0.0 for node in nx_graph.nodes()}
    
    # Assign initial sensitivities based on connected sensitive edges
    for edge_key, sensitivity in edge_sensitivities.items():
        src, dst = map(int, edge_key.split(','))
        if src in node_sensitivities and dst in node_sensitivities:
            node_sensitivities[src] += sensitivity * 0.5
            node_sensitivities[dst] += sensitivity * 0.5
    
    # If no sensitivities, use degree centrality as a starting point
    if sum(node_sensitivities.values()) == 0:
        degree_cent = nx.degree_centrality(nx_graph)
        for node, centrality in degree_cent.items():
            node_sensitivities[node] = centrality
    
    # Normalize initial sensitivities
    total = sum(node_sensitivities.values())
    if total > 0:
        for node in node_sensitivities:
            node_sensitivities[node] /= total
    
    # Iterative propagation
    for _ in range(num_iterations):
        new_sensitivities = {node: 0.0 for node in nx_graph.nodes()}
        
        # Propagation phase
        for node in nx_graph.nodes():
            neighbors = list(nx_graph.neighbors(node))
            if neighbors:
                # Distribute current node's sensitivity to neighbors
                for neighbor in neighbors:
                    new_sensitivities[neighbor] += (damping * node_sensitivities[node]) / len(neighbors)
            
            # Retain some initial sensitivity
            new_sensitivities[node] += (1 - damping) * node_sensitivities[node]
        
        # Update sensitivities
        node_sensitivities = new_sensitivities
    
    return node_sensitivities

def visualize_group_comparison_english(original_graph, extended_graph, output_path, group_type=None):
    """
    Create comparison visualization showing the difference between original and extended graphs
    with English labels for paper publication
    
    Parameters:
    - original_graph: Original PyG graph object
    - extended_graph: Extended PyG graph object
    - output_path: Output image path
    - group_type: Group type name
    """
    # Create a 3x2 subplot layout
    fig = plt.figure(figsize=(18, 16))
    
    # 1. Original graph structure (top left)
    ax1 = plt.subplot(3, 2, 1)
    nx_orig = to_networkx(original_graph, to_undirected=True)
    
    # Calculate node sensitivity for original graph
    orig_sensitivities = calculate_node_sensitivity(original_graph)
    
    # Node positions
    pos_orig = nx.spring_layout(nx_orig, seed=42)
    
    # Node colors based on sensitivity
    node_colors_orig = [orig_sensitivities[node] for node in nx_orig.nodes()]
    
    # Draw original graph
    nx.draw_networkx(
        nx_orig, 
        pos=pos_orig,
        node_color=node_colors_orig,
        node_size=300,
        cmap=plt.cm.YlOrRd,
        with_labels=True,
        font_size=10,
        font_color='black',
        font_weight='bold',
        ax=ax1
    )
    ax1.set_title("Original Graph Structure", fontsize=14)
    
    # 2. Extended graph structure (top right)
    ax2 = plt.subplot(3, 2, 2)
    nx_ext = to_networkx(extended_graph, to_undirected=True)
    
    # Calculate extended graph node sensitivity
    ext_sensitivities = calculate_node_sensitivity(extended_graph)
    
    # Node positions
    pos_ext = nx.spring_layout(nx_ext, seed=42)
    
    # Node colors based on sensitivity
    node_colors_ext = [ext_sensitivities[node] for node in nx_ext.nodes()]
    
    # Draw extended graph
    nx.draw_networkx(
        nx_ext, 
        pos=pos_ext,
        node_color=node_colors_ext,
        node_size=300,
        cmap=plt.cm.YlOrRd,
        with_labels=True,
        font_size=10,
        font_color='black',
        font_weight='bold',
        ax=ax2
    )
    ax2.set_title("Extended Graph Structure", fontsize=14)
    
    # 3. Original graph sensitivity distribution (middle left)
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(list(orig_sensitivities.values()), bins=20, color='skyblue', edgecolor='black')
    ax3.set_title("Original Graph Sensitivity Distribution", fontsize=14)
    ax3.set_xlabel("Sensitivity Value", fontsize=12)
    ax3.set_ylabel("Node Count", fontsize=12)
    
    # 4. Extended graph sensitivity distribution (middle right)
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(list(ext_sensitivities.values()), bins=20, color='salmon', edgecolor='black')
    ax4.set_title("Extended Graph Sensitivity Distribution", fontsize=14)
    ax4.set_xlabel("Sensitivity Value", fontsize=12)
    ax4.set_ylabel("Node Count", fontsize=12)
    
    # 5. Original graph statistics (bottom left)
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('off')
    
    # Calculate original graph statistics
    num_nodes_orig = nx_orig.number_of_nodes()
    num_edges_orig = nx_orig.number_of_edges()
    avg_degree_orig = sum(dict(nx_orig.degree()).values()) / num_nodes_orig if num_nodes_orig > 0 else 0
    density_orig = nx.density(nx_orig)
    
    # Display original graph statistics
    stats_text_orig = (
        f"Original Graph Statistics:\n\n"
        f"Node Count: {num_nodes_orig}\n"
        f"Edge Count: {num_edges_orig}\n"
        f"Avg Degree: {avg_degree_orig:.2f}\n"
        f"Graph Density: {density_orig:.4f}\n"
    )
    
    # Add sensitivity related statistics
    if hasattr(original_graph, 'graph_sensitivity') and original_graph.graph_sensitivity is not None:
        if hasattr(original_graph.graph_sensitivity, 'item'):
            stats_text_orig += f"Graph Sensitivity: {original_graph.graph_sensitivity.item():.4f}\n"
        else:
            stats_text_orig += f"Graph Sensitivity: {float(original_graph.graph_sensitivity):.4f}\n"
    
    if hasattr(original_graph, 'sensitive_density') and original_graph.sensitive_density is not None:
        if hasattr(original_graph.sensitive_density, 'item'):
            stats_text_orig += f"Sensitive Density: {original_graph.sensitive_density.item():.4f}\n"
        else:
            stats_text_orig += f"Sensitive Density: {float(original_graph.sensitive_density):.4f}\n"
    
    ax5.text(0.1, 0.5, stats_text_orig, fontsize=12, va='center')
    
    # 6. Extended graph statistics (bottom right)
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Calculate extended graph statistics
    num_nodes_ext = nx_ext.number_of_nodes()
    num_edges_ext = nx_ext.number_of_edges()
    avg_degree_ext = sum(dict(nx_ext.degree()).values()) / num_nodes_ext if num_nodes_ext > 0 else 0
    density_ext = nx.density(nx_ext)
    
    # Display extended graph statistics
    stats_text_ext = (
        f"Extended Graph Statistics:\n\n"
        f"Node Count: {num_nodes_ext}\n"
        f"Edge Count: {num_edges_ext}\n"
        f"Avg Degree: {avg_degree_ext:.2f}\n"
        f"Graph Density: {density_ext:.4f}\n"
    )
    
    # Add sensitivity related statistics
    if hasattr(extended_graph, 'graph_sensitivity') and extended_graph.graph_sensitivity is not None:
        if hasattr(extended_graph.graph_sensitivity, 'item'):
            stats_text_ext += f"Graph Sensitivity: {extended_graph.graph_sensitivity.item():.4f}\n"
        else:
            stats_text_ext += f"Graph Sensitivity: {float(extended_graph.graph_sensitivity):.4f}\n"
    
    if hasattr(extended_graph, 'sensitive_density') and extended_graph.sensitive_density is not None:
        if hasattr(extended_graph.sensitive_density, 'item'):
            stats_text_ext += f"Sensitive Density: {extended_graph.sensitive_density.item():.4f}\n"
        else:
            stats_text_ext += f"Sensitive Density: {float(extended_graph.sensitive_density):.4f}\n"
    
    ax6.text(0.1, 0.5, stats_text_ext, fontsize=12, va='center')
    
    # Set main title
    group_name = group_type.capitalize() if group_type else "Group"
    fig.suptitle(f"{group_name} Graph Structure Extension Comparison", fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_paper_visualization_english(graph, output_path, title=None):
    """
    Create high-quality graph structure and sensitivity visualization with English labels
    
    Parameters:
    - graph: PyG graph data object
    - output_path: Output image path
    - title: Optional title
    """
    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Graph structure visualization (top left)
    ax1 = plt.subplot(2, 2, 1)
    nx_graph = to_networkx(graph, to_undirected=True)
    
    # Calculate node sensitivity
    node_sensitivities = calculate_node_sensitivity(graph)
    
    # Node positions
    pos = nx.spring_layout(nx_graph, seed=42)
    
    # Node colors based on sensitivity
    node_colors = [node_sensitivities[node] for node in nx_graph.nodes()]
    
    # Draw graph
    nx.draw_networkx(
        nx_graph, 
        pos=pos,
        node_color=node_colors,
        node_size=300,
        cmap=plt.cm.YlOrRd,
        with_labels=True,
        font_size=10,
        font_color='black',
        font_weight='bold',
        ax=ax1
    )
    ax1.set_title("Graph Structure with Node Sensitivity", fontsize=14)
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Node Sensitivity')
    
    # 2. Sensitivity heat map (top right)
    ax2 = plt.subplot(2, 2, 2)
    
    # Create adjacency matrix with sensitivity
    num_nodes = len(nx_graph.nodes())
    adj_matrix = np.zeros((num_nodes, num_nodes))
    sensitivity_matrix = np.zeros((num_nodes, num_nodes))
    
    edge_index = graph.edge_index.numpy()
    if hasattr(graph, 'sensitive_edge_mask') and graph.sensitive_edge_mask is not None:
        sensitive_edge_mask = graph.sensitive_edge_mask.numpy()
    else:
        sensitive_edge_mask = np.ones(edge_index.shape[1])
    
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        adj_matrix[source][target] = 1
        adj_matrix[target][source] = 1  # Undirected graph
        sensitivity_matrix[source][target] = sensitive_edge_mask[i]
        sensitivity_matrix[target][source] = sensitive_edge_mask[i]  # Undirected graph
    
    # Draw heat map
    im = ax2.imshow(sensitivity_matrix, cmap='YlOrRd')
    ax2.set_title("Sensitivity Heat Map", fontsize=14)
    plt.colorbar(im, ax=ax2)
    
    # 3. Sensitivity distribution histogram (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(list(node_sensitivities.values()), bins=20, color='skyblue', edgecolor='black')
    ax3.set_title("Node Sensitivity Distribution", fontsize=14)
    ax3.set_xlabel("Sensitivity Value", fontsize=12)
    ax3.set_ylabel("Node Count", fontsize=12)
    
    # 4. Graph statistics (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate graph statistics
    num_nodes = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    avg_degree = sum(dict(nx_graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
    density = nx.density(nx_graph)
    
    # Display statistics
    stats_text = (
        f"Graph Statistics:\n\n"
        f"Node Count: {num_nodes}\n"
        f"Edge Count: {num_edges}\n"
        f"Avg Degree: {avg_degree:.2f}\n"
        f"Graph Density: {density:.4f}\n"
    )
    
    # Add sensitivity related statistics
    if hasattr(graph, 'graph_sensitivity') and graph.graph_sensitivity is not None:
        if hasattr(graph.graph_sensitivity, 'item'):
            stats_text += f"Graph Sensitivity: {graph.graph_sensitivity.item():.4f}\n"
        else:
            stats_text += f"Graph Sensitivity: {float(graph.graph_sensitivity):.4f}\n"
    
    if hasattr(graph, 'sensitive_density') and graph.sensitive_density is not None:
        if hasattr(graph.sensitive_density, 'item'):
            stats_text += f"Sensitive Density: {graph.sensitive_density.item():.4f}\n"
        else:
            stats_text += f"Sensitive Density: {float(graph.sensitive_density):.4f}\n"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    # Set main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_sensitive_edges(graph, output_path, title="Edge Sensitivity Analysis", semantic_sensitivities=None):
    """Create visualization focused on edge sensitivity distribution with semantic comparison
    
    Args:
        graph: PyG graph data object
        output_path: Output image path
        title: Title for the visualization
        semantic_sensitivities: Optional semantic sensitivities after injection
    """
    # Create figure with subplots
    if semantic_sensitivities is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    
    # Convert to NetworkX for visualization
    nx_graph = to_networkx(graph, to_undirected=False)
    
    # Get edge sensitivities
    edge_sensitivities = {}
    if hasattr(graph, 'edge_sensitivity'):
        edge_sens = graph.edge_sensitivity.cpu().numpy()
        edge_index = graph.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            edge_sensitivities[(edge_index[0, i], edge_index[1, i])] = float(edge_sens[i])
    
    # Process triplets if available
    if hasattr(graph, 'triplets') and graph.triplets is not None:
        triplets = graph.triplets.cpu().numpy()
        edge_index_list = []
        for triplet in triplets:
            edge_index_list.extend([
                [triplet[0], triplet[1]],
                [triplet[1], triplet[2]],
                [triplet[0], triplet[2]]
            ])
        edge_index = np.array(edge_index_list).T
        
        # Update edge sensitivities for triplets
        edge_sensitivities = {}
        for i in range(edge_index.shape[1]):
            edge_sensitivities[(edge_index[0, i], edge_index[1, i])] = float(edge_sens[i % len(edge_sens)])
    
    # Compute layout once and reuse
    pos = nx.spring_layout(nx_graph, k=2, iterations=50)
    
    # Draw structural sensitivity graph
    ax1.set_title('Structural Sensitivity', fontsize=14)
    
    # Draw edges with arrows
    edge_colors = []
    edge_widths = []
    edges = []
    for (u, v), sens in edge_sensitivities.items():
        edges.append((u, v))
        edge_colors.append(sens)
        edge_widths.append(1 + 2 * sens)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(
        nx_graph, pos,
        ax=ax1,
        edgelist=edges,
        edge_color=edge_colors,
        width=edge_widths,
        edge_cmap=plt.cm.YlOrRd,
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.2'
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph, pos,
        ax=ax1,
        node_color='lightblue',
        node_size=1000,
        alpha=0.8
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        nx_graph, pos,
        ax=ax1,
        font_size=12,
        font_weight='bold'
    )
    
    # Add edge labels with sensitivities
    edge_labels = {(u, v): f"{sens:.2f}" for (u, v), sens in edge_sensitivities.items()}
    nx.draw_networkx_edge_labels(
        nx_graph, pos,
        ax=ax1,
        edge_labels=edge_labels,
        font_size=10
    )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm, ax=ax1, label='Structural Sensitivity')
    
    if semantic_sensitivities is not None:
        # Draw semantic sensitivity graph
        ax2.set_title('After Semantic Injection', fontsize=14)
        
        # Draw edges with arrows
        semantic_colors = []
        semantic_widths = []
        for (u, v) in edges:
            sens = semantic_sensitivities.get((u, v), 0)
            semantic_colors.append(sens)
            semantic_widths.append(1 + 2 * sens)
        
        nx.draw_networkx_edges(
            nx_graph, pos,
            ax=ax2,
            edgelist=edges,
            edge_color=semantic_colors,
            width=semantic_widths,
            edge_cmap=plt.cm.YlOrRd,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.2'
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            nx_graph, pos,
            ax=ax2,
            node_color='lightblue',
            node_size=1000,
            alpha=0.8
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            nx_graph, pos,
            ax=ax2,
            font_size=12,
            font_weight='bold'
        )
        
        # Add edge labels with sensitivities and changes
        edge_labels = {}
        for (u, v) in edges:
            old_sens = edge_sensitivities.get((u, v), 0)
            new_sens = semantic_sensitivities.get((u, v), 0)
            change = new_sens - old_sens
            sign = '+' if change >= 0 else ''
            edge_labels[(u, v)] = f"{new_sens:.2f}\n({sign}{change:.2f})"
        
        nx.draw_networkx_edge_labels(
            nx_graph, pos,
            ax=ax2,
            edge_labels=edge_labels,
            font_size=10
        )
        
        # Add colorbar
        plt.colorbar(sm, ax=ax2, label='Semantic Sensitivity')
        
        # Add statistics
        stats_text = (
            f"Avg Change: {np.mean([s - edge_sensitivities.get(e, 0) for e, s in semantic_sensitivities.items()]):.3f}\n"
            f"Max Change: {np.max([s - edge_sensitivities.get(e, 0) for e, s in semantic_sensitivities.items()]):.3f}\n"
            f"Min Change: {np.min([s - edge_sensitivities.get(e, 0) for e, s in semantic_sensitivities.items()]):.3f}"
        )
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_sensitivity_distribution(sensitivities: np.ndarray,
                                    title: str = "Sensitivity Distribution",
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> None:
    """
    Visualize sensitivity distribution
    
    Args:
        sensitivities: Sensitivity array
        title: Plot title
        save_path: Save path (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=sensitivities.flatten(), kde=True)
    plt.title(title)
    plt.xlabel("sensitivity")
    plt.ylabel("frequency")
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def visualize_graph_sensitivity(graph: nx.Graph,
                              node_sensitivities: np.ndarray,
                              edge_sensitivities: Optional[np.ndarray] = None,
                              title: str = "Graph Sensitivity Analysis",
                              save_path: Optional[str] = None,
                              show: bool = True) -> None:
    """
    Visualize node and edge sensitivities of a graph
    
    Args:
        graph: NetworkX graph object
        node_sensitivities: Node sensitivity array
        edge_sensitivities: Edge sensitivity array (optional)
        title: Plot title
        save_path: Save path (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos,
                          node_color=node_sensitivities,
                          node_size=500,
                          cmap=plt.cm.RdYlBu,
                          alpha=0.7)
    
    # Draw edges
    if edge_sensitivities is not None:
        edges = list(graph.edges())
        edge_colors = [edge_sensitivities[i] for i in range(len(edges))]
        nx.draw_networkx_edges(graph, pos,
                             edge_color=edge_colors,
                             edge_cmap=plt.cm.RdYlBu,
                             width=2,
                             alpha=0.7)
    else:
        nx.draw_networkx_edges(graph, pos)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu)
    sm.set_array([])
    plt.colorbar(sm, label="sensitivity")
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def visualize_path_sensitivity(graph: nx.Graph,
                             paths: List[List[int]],
                             path_sensitivities: np.ndarray,
                             title: str = "Path Sensitivity Analysis",
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
    """
    Visualize path sensitivity
    
    Args:
        graph: NetworkX graph object
        paths: List of paths
        path_sensitivities: Path sensitivity array
        title: Plot title
        save_path: Save path (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot path sensitivity distribution
    sns.histplot(data=path_sensitivities, kde=True, ax=ax1)
    ax1.set_title("Path Sensitivity Distribution")
    ax1.set_xlabel("Sensitivity")
    ax1.set_ylabel("Frequency")
    
    # Plot relationship between path length and sensitivity
    path_lengths = [len(path) for path in paths]
    ax2.scatter(path_lengths, path_sensitivities)
    ax2.set_title("Path Length vs Sensitivity")
    ax2.set_xlabel("Path Length")
    ax2.set_ylabel("Sensitivity")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def visualize_sensitivity_comparison(original_sensitivities: np.ndarray,
                                   modified_sensitivities: np.ndarray,
                                   labels: List[str] = ["Original", "Modified"],
                                   title: str = "Sensitivity Comparison Analysis",
                                   save_path: Optional[str] = None,
                                   show: bool = True) -> None:
    """
    Visualize sensitivity comparison
    
    Args:
        original_sensitivities: Original sensitivity array
        modified_sensitivities: Modified sensitivity array
        labels: Label list
        title: Plot title
        save_path: Save path (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    plt.boxplot([original_sensitivities.flatten(),
                 modified_sensitivities.flatten()],
                labels=labels)
    
    plt.title(title)
    plt.ylabel("Sensitivity")
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_sensitivity_heatmap(sensitivity_matrix: np.ndarray,
                           x_labels: Optional[List[str]] = None,
                           y_labels: Optional[List[str]] = None,
                           title: str = "Sensitivity Heatmap",
                           save_path: Optional[str] = None,
                           show: bool = True) -> None:
    """
    Plot sensitivity heatmap
    
    Args:
        sensitivity_matrix: Sensitivity matrix
        x_labels: x-axis labels (optional)
        y_labels: y-axis labels (optional)
        title: Plot title
        save_path: Save path (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(sensitivity_matrix,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cmap='RdYlBu',
                center=0,
                annot=True,
                fmt='.2f')
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def create_interactive_sensitivity_plot(graph: nx.Graph,
                                     node_sensitivities: np.ndarray,
                                     edge_sensitivities: Optional[np.ndarray] = None,
                                     title: str = "Interactive Sensitivity Visualization") -> go.Figure:
    """
    Create interactive sensitivity visualization
    
    Args:
        graph: NetworkX graph object
        node_sensitivities: Node sensitivity array
        edge_sensitivities: Edge sensitivity array (optional)
        title: Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Get node positions
    pos = nx.spring_layout(graph, dim=3)
    
    # Create node trace
    node_trace = go.Scatter3d(
        x=[pos[node][0] for node in graph.nodes()],
        y=[pos[node][1] for node in graph.nodes()],
        z=[pos[node][2] for node in graph.nodes()],
        mode='markers',
        marker=dict(
            size=10,
            color=node_sensitivities,
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title="Node Sensitivity")
        ),
        text=[f"Node {node}<br>Sensitivity: {sens:.3f}"
              for node, sens in zip(graph.nodes(), node_sensitivities)],
        hoverinfo='text'
    )
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(
            color='#888' if edge_sensitivities is None else edge_sensitivities,
            width=1
        ),
        hoverinfo='none'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        )
    )
    
    return fig

# Enhanced edge sensitivity visualization function
def visualize_enhanced_edge_sensitivity(graph: nx.Graph,
                                        edge_sensitivities: np.ndarray,
                                        node_positions=None,
                                        title: str = "Enhanced Edge Sensitivity Analysis",
                                        highlight_threshold: float = 0.75,
                                        save_path: Optional[str] = None,
                                        show: bool = True) -> plt.Figure:
    """
    Visualize edge-level sensitivity with highlighting of high sensitivity edges
    
    Args:
        graph: Input graph
        edge_sensitivities: Edge sensitivity array
        node_positions: Node positions, if None will be computed automatically
        title: Plot title
        highlight_threshold: Sensitivity threshold for highlighting (percentile, e.g. 0.75 means top 25% edges are highlighted)
        save_path: Save path
        show: Whether to display plot
        
    Returns:
        plt.Figure: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Ensure edge sensitivities is numpy array
    if not isinstance(edge_sensitivities, np.ndarray):
        edge_sensitivities = np.array(edge_sensitivities)
    
    # If no node positions provided, compute using spring layout
    if node_positions is None:
        node_positions = nx.spring_layout(graph, seed=42)
    
    # Calculate highlight threshold
    high_sensitivity_threshold = np.percentile(edge_sensitivities, highlight_threshold * 100)
    
    # Create edge list and corresponding sensitivity list
    edges = list(graph.edges())
    edge_colors = []
    edge_widths = []
    
    # Assign colors and widths for each edge
    for i, (u, v) in enumerate(edges):
        if i < len(edge_sensitivities):
            sensitivity = edge_sensitivities[i]
            # Normalize sensitivity to 0-1
            norm_sensitivity = (sensitivity - np.min(edge_sensitivities)) / (np.max(edge_sensitivities) - np.min(edge_sensitivities) + 1e-10)
            edge_colors.append(norm_sensitivity)
            
            # High sensitivity edges get larger width
            if sensitivity >= high_sensitivity_threshold:
                edge_widths.append(3.0 + 2.0 * norm_sensitivity)  # 3-5 width
            else:
                edge_widths.append(1.0 + 1.0 * norm_sensitivity)  # 1-2 width
        else:
            edge_colors.append(0.0)
            edge_widths.append(1.0)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, node_positions, node_size=500, node_color='lightblue', alpha=0.8, ax=ax)
    
    # Draw edges with sensitivity colors
    edges = nx.draw_networkx_edges(
        graph, node_positions, 
        edge_color=edge_colors,
        width=edge_widths,
        edge_cmap=plt.cm.viridis,
        edge_vmin=0.0,
        edge_vmax=1.0,
        arrows=True,
        arrowsize=15,
        ax=ax
    )
    
    # Draw node labels
    nx.draw_networkx_labels(graph, node_positions, font_size=10, font_weight='bold', ax=ax)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Edge Sensitivity', shrink=0.8)
    
    # Add highlight threshold marker
    norm_threshold = (high_sensitivity_threshold - np.min(edge_sensitivities)) / (np.max(edge_sensitivities) - np.min(edge_sensitivities) + 1e-10)
    cbar.ax.axhline(y=norm_threshold, color='r', linestyle='--')
    cbar.ax.text(0.5, norm_threshold, f"High Sensitivity Threshold ({highlight_threshold:.2f})",
                 va='center', ha='center', color='r', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Set title and axis
    ax.set_title(title, fontsize=15)
    ax.axis('off')
    
    # Add legend for highlighted edges
    high_sens_patch = mpatches.Patch(color='yellow', label='High Sensitivity Edges')
    plt.legend(handles=[high_sens_patch], loc='lower right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save and show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

# Path sensitivity and QA analysis function
def analyze_path_sensitivity_qa(graph: nx.Graph,
                              paths: List[List[int]],
                              path_sensitivities: np.ndarray,
                              predictions: List[float],
                              ground_truth: List[float],
                              high_sensitivity_threshold: float = 0.75,
                              save_path: Optional[str] = None,
                              show: bool = True) -> Dict[str, Any]:
    """Analyze relationship between path sensitivity and QA performance
    
    Args:
        graph: Input graph
        paths: Path list
        path_sensitivities: Path sensitivity array
        predictions: Predicted values
        ground_truth: Ground truth values
        high_sensitivity_threshold: High sensitivity threshold (percentile)
        save_path: Save path
        show: Whether to display plot
        
    Returns:
        Dict: Analysis results
    """
    # Calculate initial sensitivities based on node properties
    node_degrees = dict(graph.degree())
    max_degree = max(node_degrees.values())
    node_centrality = nx.betweenness_centrality(graph)
    
    # Initialize edge sensitivities based on structural properties
    edge_sensitivities = {}
    for u, v in graph.edges():
        # Combine degree and centrality information
        degree_factor = (node_degrees[u] + node_degrees[v]) / (2 * max_degree)
        centrality_factor = (node_centrality[u] + node_centrality[v]) / 2
        
        # Calculate initial sensitivity
        initial_sens = 0.3 * degree_factor + 0.7 * centrality_factor
        edge_sensitivities[(u, v)] = initial_sens
    
    # Calculate path lengths
    path_lengths = [len(p) for p in paths]
    
    # Calculate prediction errors
    errors = np.abs(np.array(predictions) - np.array(ground_truth))
    
    # Calculate high sensitivity edge ratios for each path
    high_sensitivity_ratios = []
    for path in paths:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        high_sens_edges = [e for e in path_edges if edge_sensitivities.get(e, 0) > high_sensitivity_threshold]
        ratio = len(high_sens_edges) / max(1, len(path_edges))
        high_sensitivity_ratios.append(ratio)
    
    # Visualization
    if show or save_path:
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Path length vs prediction error
        ax = axs[0, 0]
        if len(path_lengths) > 1:
            scatter = ax.scatter(path_lengths, errors, alpha=0.7, c=path_sensitivities, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Path Sensitivity')
            
            # Add trend line
            z = np.polyfit(path_lengths, errors, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(path_lengths), max(path_lengths), 100)
            ax.plot(x_range, p(x_range), 'r--', 
                   label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data (n<2)', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Path Length')
        ax.set_ylabel('Prediction Error')
        ax.set_title(f'Path Length vs Prediction Error (n={len(paths)})')
        ax.grid(True, alpha=0.3)
        
        # 2. High sensitivity edge ratio vs prediction error
        ax = axs[0, 1]
        if len(high_sensitivity_ratios) > 1:
            scatter = ax.scatter(high_sensitivity_ratios, errors, alpha=0.7, c=path_sensitivities, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label='Path Sensitivity')
        else:
            ax.text(0.5, 0.5, 'Insufficient data (n<2)', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('High Sensitivity Edge Ratio')
        ax.set_ylabel('Prediction Error')
        ax.set_title('Sensitivity Ratio vs Error')
        ax.grid(True, alpha=0.3)
        
        # 3. Path sensitivity distribution
        ax = axs[1, 0]
        if len(path_sensitivities) > 1:
            sns.histplot(path_sensitivities, bins=20, ax=ax, kde=True)
            ax.axvline(high_sensitivity_threshold, color='r', linestyle='--',
                      label=f'Threshold ({high_sensitivity_threshold:.2f})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data (n<2)', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Path Sensitivity')
        ax.set_ylabel('Count')
        ax.set_title('Path Sensitivity Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Edge sensitivity visualization
        ax = axs[1, 1]
        pos = nx.spring_layout(graph)
        
        # Draw edges with sensitivity colors
        edges = nx.draw_networkx_edges(
            graph, pos,
            ax=ax,
            edge_color=list(edge_sensitivities.values()),
            edge_cmap=plt.cm.YlOrRd,
            width=2,
            alpha=0.7
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            ax=ax,
            node_color='lightblue',
            node_size=500,
            alpha=0.8
        )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm, ax=ax, label='Edge Sensitivity')
        
        ax.set_title('Edge Sensitivity Map')
        ax.axis('off')
        
        # Add statistics
        stats_text = (
            f"Avg Error: {np.mean(errors):.4f}\n"
            f"Avg Path Length: {np.mean(path_lengths):.2f}\n"
            f"Avg Sensitivity: {np.mean(path_sensitivities):.4f}\n"
            f"High Sens Ratio: {np.mean(high_sensitivity_ratios):.4f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # Calculate statistics
    results = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'mean_path_length': float(np.mean(path_lengths)),
        'mean_sensitivity': float(np.mean(path_sensitivities)),
        'mean_high_sens_ratio': float(np.mean(high_sensitivity_ratios)),
        'correlation_length_error': float(np.corrcoef(path_lengths, errors)[0, 1] if len(path_lengths) > 1 else 0),
        'correlation_sens_error': float(np.corrcoef(path_sensitivities, errors)[0, 1] if len(path_sensitivities) > 1 else 0)
    }
    
    return results

# Semantic flow visualization function
def visualize_semantic_flow(original_sensitivities: np.ndarray,
                          semantic_enhanced_sensitivities: np.ndarray,
                          graph: Optional[nx.Graph] = None,
                          node_mapping: Optional[Dict[int, str]] = None,
                          title: str = "Sensitivity Changes Before and After Semantic Injection",
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """
    Visualize sensitivity flow changes before and after semantic injection
    
    Args:
        original_sensitivities: Original structural sensitivities
        semantic_enhanced_sensitivities: Semantically enhanced sensitivities
        graph: Optional graph object for node mapping
        node_mapping: Node ID to name mapping
        title: Plot title
        save_path: Save path
        show: Whether to display plot
        
    Returns:
        plt.Figure: matplotlib figure object
    """
    # Ensure inputs are numpy arrays
    if not isinstance(original_sensitivities, np.ndarray):
        original_sensitivities = np.array(original_sensitivities)
    if not isinstance(semantic_enhanced_sensitivities, np.ndarray):
        semantic_enhanced_sensitivities = np.array(semantic_enhanced_sensitivities)
    
    # Calculate sensitivity changes
    sensitivity_change = semantic_enhanced_sensitivities - original_sensitivities
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sensitivity heatmap comparison
    if graph is not None and len(original_sensitivities) == len(graph.nodes()):
        # Use graph structure for visualization
        
        # Calculate node positions
        pos = nx.spring_layout(graph, seed=42)
        
        # Draw original sensitivities
        nx.draw_networkx_nodes(graph, pos, node_color=original_sensitivities, 
                             cmap=plt.cm.viridis, node_size=500, alpha=0.8, ax=axs[0, 0])
        nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=axs[0, 0])
        if node_mapping:
            labels = {n: node_mapping.get(n, str(n)) for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=axs[0, 0])
        else:
            nx.draw_networkx_labels(graph, pos, font_size=8, ax=axs[0, 0])
        axs[0, 0].set_title('Original Structural Sensitivity')
        axs[0, 0].axis('off')
        
        # Draw enhanced sensitivities
        nx.draw_networkx_nodes(graph, pos, node_color=semantic_enhanced_sensitivities, 
                             cmap=plt.cm.viridis, node_size=500, alpha=0.8, ax=axs[0, 1])
        nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=axs[0, 1])
        if node_mapping:
            labels = {n: node_mapping.get(n, str(n)) for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=axs[0, 1])
        else:
            nx.draw_networkx_labels(graph, pos, font_size=8, ax=axs[0, 1])
        axs[0, 1].set_title('Semantically Enhanced Sensitivity')
        axs[0, 1].axis('off')
        
        # Draw change heatmap
        nx.draw_networkx_nodes(graph, pos, node_color=sensitivity_change, 
                             cmap=plt.cm.coolwarm, node_size=500, alpha=0.8, ax=axs[1, 0])
        nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=axs[1, 0])
        if node_mapping:
            labels = {n: node_mapping.get(n, str(n)) for n in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=axs[1, 0])
        else:
            nx.draw_networkx_labels(graph, pos, font_size=8, ax=axs[1, 0])
        axs[1, 0].set_title('Sensitivity Changes (Red=Increase, Blue=Decrease)')
        axs[1, 0].axis('off')
        
    else:
        # Use heatmap visualization
        n = int(np.sqrt(len(original_sensitivities))) + 1
        orig_reshaped = original_sensitivities[:n*n].reshape((n, n))
        enhanced_reshaped = semantic_enhanced_sensitivities[:n*n].reshape((n, n))
        change_reshaped = sensitivity_change[:n*n].reshape((n, n))
        
        sns.heatmap(orig_reshaped, ax=axs[0, 0], cmap='viridis', annot=False)
        axs[0, 0].set_title('Original Structural Sensitivity')
        
        sns.heatmap(enhanced_reshaped, ax=axs[0, 1], cmap='viridis', annot=False)
        axs[0, 1].set_title('Semantically Enhanced Sensitivity')
        
        sns.heatmap(change_reshaped, ax=axs[1, 0], cmap='coolwarm', center=0, annot=False)
        axs[1, 0].set_title('Sensitivity Changes (Red=Increase, Blue=Decrease)')
    
    # 2. Scatter comparison plot
    axs[1, 1].scatter(original_sensitivities, semantic_enhanced_sensitivities, alpha=0.7)
    axs[1, 1].plot([0, 1], [0, 1], 'r--')  # Diagonal line
    axs[1, 1].set_xlabel('Original Structural Sensitivity')
    axs[1, 1].set_ylabel('Semantically Enhanced Sensitivity')
    axs[1, 1].set_title('Sensitivity Comparison Scatter Plot')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save and show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

# Calculate high sensitivity edge ratio in prediction paths
def calculate_high_sensitivity_edge_ratio(paths: List[List[int]],
                                        edge_sensitivities: Dict[Tuple[int, int], float],
                                        high_sensitivity_threshold: float = 0.75) -> Dict[str, Any]:
    """
    Calculate ratio of high sensitivity edges in prediction paths
    
    Args:
        paths: Path list
        edge_sensitivities: Edge sensitivity dictionary, keys are edge tuples, values are sensitivities
        high_sensitivity_threshold: High sensitivity threshold (percentile)
        
    Returns:
        Dict: Dictionary containing analysis results
    """
    # Determine high sensitivity edges
    sensitivity_values = list(edge_sensitivities.values())
    threshold = np.percentile(sensitivity_values, high_sensitivity_threshold * 100)
    high_sensitivity_edges = {e for e, v in edge_sensitivities.items() if v >= threshold}
    
    # Calculate ratio of high sensitivity edges for each path
    path_stats = []
    for i, path in enumerate(paths):
        # Build path edges
        path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
        
        # Calculate number and ratio of high sensitivity edges
        high_sens_edges = [e for e in path_edges if e in high_sensitivity_edges]
        
        path_stats.append({
            "path_id": i,
            "path_length": len(path),
            "edge_count": len(path_edges),
            "high_sensitivity_edge_count": len(high_sens_edges),
            "high_sensitivity_ratio": len(high_sens_edges) / max(1, len(path_edges))
        })
    
    # Calculate statistics
    high_sensitivity_ratios = [stat["high_sensitivity_ratio"] for stat in path_stats]
    
    results = {
        "path_stats": path_stats,
        "average_high_sensitivity_ratio": np.mean(high_sensitivity_ratios),
        "median_high_sensitivity_ratio": np.median(high_sensitivity_ratios),
        "threshold": threshold,
        "high_sensitivity_edge_count": len(high_sensitivity_edges),
        "total_edge_count": len(edge_sensitivities)
    }
    
    return results

# Error-path length correlation analysis
def analyze_error_path_length_correlation(paths: List[List[int]],
                                        predictions: List[float],
                                        ground_truth: List[float],
                                        save_path: Optional[str] = None,
                                        show: bool = True) -> Dict[str, Any]:
    """
    Analyze correlation between prediction error and path length
    
    Args:
        paths: Path list
        predictions: Prediction list
        ground_truth: Ground truth list
        save_path: Save path
        show: Whether to display plot
        
    Returns:
        Dict: Dictionary containing analysis results
    """
    # Ensure input lengths match
    assert len(paths) == len(predictions) == len(ground_truth), "Input array lengths must match"
    
    # Calculate path lengths
    path_lengths = [len(p) for p in paths]
    
    # Calculate prediction errors
    errors = np.abs(np.array(predictions) - np.array(ground_truth))
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(path_lengths, errors)[0, 1]
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(path_lengths, errors)
    
    results = {
        "correlation": correlation,
        "r_squared": r_value**2,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "std_error": std_err,
        "path_lengths": path_lengths,
        "errors": errors.tolist()
    }
    
    # Visualization
    if show or save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw scatter plot
        ax.scatter(path_lengths, errors, alpha=0.7)
        
        # Draw trend line
        x_range = np.linspace(min(path_lengths), max(path_lengths), 100)
        ax.plot(x_range, intercept + slope * x_range, 'r--', 
               label=f'y = {slope:.4f}x + {intercept:.4f} (R² = {r_value**2:.4f})')
        
        # Set labels and title
        ax.set_xlabel('Path Length')
        ax.set_ylabel('Prediction Error')
        ax.set_title(f'Error vs Path Length Correlation Analysis (Correlation = {correlation:.4f})')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        results["figure"] = fig
    
    return results 

class TrainingVisualizer:
    """Training process visualizer"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'structure_score': [],
            'reasoning_score': []
        }
    
    def update(self, metrics):
        """Update metrics"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def plot_metrics(self):
        """Plot training metrics"""
        plt.figure(figsize=(12, 8))
        
        # Plot loss curves
        plt.subplot(2, 1, 1)
        if self.metrics['train_loss']:
            plt.plot(self.metrics['train_loss'], label='Train Loss')
        if self.metrics['val_loss']:
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot score curves
        plt.subplot(2, 1, 2)
        if self.metrics['structure_score']:
            plt.plot(self.metrics['structure_score'], label='Structure Score')
        if self.metrics['reasoning_score']:
            plt.plot(self.metrics['reasoning_score'], label='Reasoning Score')
        plt.title('Model Performance Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()

class ProbabilityVisualizer:
    """Probability distribution visualizer"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_probability_distribution(self, probabilities, title="Probability Distribution"):
        """Plot probability distribution"""
        plt.figure(figsize=(10, 6))
        sns.histplot(probabilities, bins=50, kde=True)
        plt.title(title)
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'probability_distribution.png'))
        plt.close()
    
    def plot_probability_comparison(self, prob_dict, title="Probability Comparison"):
        """Plot probability comparison"""
        plt.figure(figsize=(12, 6))
        
        for label, probs in prob_dict.items():
            sns.kdeplot(probs, label=label)
        
        plt.title(title)
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'probability_comparison.png'))
        plt.close()
    
    def plot_probability_heatmap(self, prob_matrix, x_labels=None, y_labels=None, title="Probability Heatmap"):
        """Plot probability heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(prob_matrix, xticklabels=x_labels, yticklabels=y_labels, 
                   cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, 'probability_heatmap.png'))
        plt.close()

def visualize_training_metrics(metrics, output_dir, title="Training Metrics"):
    """
    Visualize various metrics during training process
    
    Args:
        metrics (dict): Dictionary containing various metrics
        output_dir (str): Output directory
        title (str): Chart title
    """
    plt.figure(figsize=(12, 8))
    
    # Determine the number of metrics to plot
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    for i, (metric_name, metric_values) in enumerate(metrics.items(), 1):
        plt.subplot(n_rows, n_cols, i)
        plt.plot(metric_values, marker='o')
        plt.title(f'{metric_name} over Time')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def visualize_path_sensitivity_comparison(original_paths, modified_paths, output_dir, title="Path Sensitivity Comparison"):
    """
    Compare sensitivity of original paths and modified paths
    
    Args:
        original_paths (list): List of sensitivity values for original paths
        modified_paths (list): List of sensitivity values for modified paths
        output_dir (str): Output directory
        title (str): Chart title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot density graphs
    sns.kdeplot(original_paths, label='Original Paths', color='blue')
    sns.kdeplot(modified_paths, label='Modified Paths', color='red')
    
    plt.title(title)
    plt.xlabel('Sensitivity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'path_sensitivity_comparison.png'))
    plt.close()

def visualize_performance_radar(metrics, output_dir, title="Model Performance Radar"):
    """
    Use radar chart to display various performance metrics
    
    Args:
        metrics (dict): Dictionary containing various performance metrics
        output_dir (str): Output directory
        title (str): Chart title
    """
    # Prepare data
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Calculate angles
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    # Close the figure
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    
    # Plot radar chart
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    
    # Set tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])
    
    plt.title(title)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'performance_radar.png'))
    plt.close()

def visualize_edge_sensitivity(edge_sensitivities: np.ndarray,
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
    """Visualize edge sensitivity distribution
    
    Args:
        edge_sensitivities: Edge sensitivity array
        save_path: Save path
        show: Whether to display the image
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(edge_sensitivities, bins=50, alpha=0.7, color='skyblue')
    plt.title('Edge Sensitivity Distribution')
    plt.xlabel('Sensitivity')
    plt.ylabel('Count')
    
    # Add statistics
    mean = np.mean(edge_sensitivities)
    std = np.std(edge_sensitivities)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
    plt.text(mean, plt.ylim()[1]*0.9, f'Mean: {mean:.3f}\nStd: {std:.3f}', 
             horizontalalignment='center')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()

def visualize_feature_space(features: np.ndarray,
                          predictions: np.ndarray,
                          save_path: Optional[str] = None) -> None:
    """可视化特征空间
    
    Args:
        features: 特征数组
        predictions: 预测值数组
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    if len(features) < 3:
        # 如果样本太少，直接绘制散点图
        plt.scatter(features[:, 0], features[:, 1], c=predictions, cmap='viridis')
        plt.colorbar(label='Prediction Value')
        plt.title('Feature Space (First 2 Dimensions)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    else:
        # 使用t-SNE降维，调整perplexity
        perplexity = min(30, len(features) - 1)  # 确保perplexity小于样本数
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        try:
            features_2d = tsne.fit_transform(features)
            
            # 绘制散点图
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=predictions, cmap='viridis')
            plt.colorbar(scatter, label='Prediction Value')
            plt.title('t-SNE Feature Space Visualization')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
        except Exception as e:
            print(f"Warning: t-SNE failed ({str(e)}), using PCA instead")
            # 如果t-SNE失败，使用PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=predictions, cmap='viridis')
            plt.colorbar(scatter, label='Prediction Value')
            plt.title('PCA Feature Space Visualization')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
    
    # 添加统计信息
    stats_text = (
        f"n_samples: {len(features)}\n"
        f"n_features: {features.shape[1]}\n"
        f"pred_mean: {predictions.mean():.4f}\n"
        f"pred_std: {predictions.std():.4f}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def visualize_performance_metrics(predictions: np.ndarray,
                                targets: np.ndarray,
                                save_path: Optional[str] = None,
                                show: bool = True) -> None:
    """可视化性能指标
    
    Args:
        predictions: 预测值数组
        targets: 目标值数组
        save_path: 保存路径
        show: 是否显示图像
    """
    # 计算性能指标
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    r2 = 1 - np.sum((predictions - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    # 创建雷达图
    metrics = ['MSE', 'MAE', 'R2']
    values = [mse, mae, r2]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Model Performance Metrics')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()

def calculate_edge_sensitivity(model, batch, prior_sensitivity=None, temperature=1.0):
    """基于后验贝叶斯的敏感度传播
    
    Args:
        model: 模型对象
        batch: 批次数据
        prior_sensitivity: 先验敏感度
        temperature: 温度参数，控制后验分布的平滑程度
        
    Returns:
        edge_sensitivities: 边敏感度字典
    """
    model.eval()
    with torch.no_grad():
        # 获取模型输出
        output = model(batch)
        logits = output['logits'] if isinstance(output, dict) else output
        
        # 计算后验概率
        posterior = torch.softmax(logits / temperature, dim=-1)
        
        # 获取边索引
        edge_index = batch.edge_index
        
        # 初始化边敏感度
        edge_sensitivities = {}
        
        # 如果没有先验敏感度，使用均匀分布
        if prior_sensitivity is None:
            prior_sensitivity = torch.ones_like(posterior) / posterior.shape[-1]
        
        # 计算每条边的后验敏感度
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            # 计算KL散度作为敏感度
            kl_div = torch.sum(posterior * torch.log(posterior / (prior_sensitivity + 1e-10)))
            sensitivity = torch.exp(-kl_div)  # 使用负KL散度的指数作为敏感度
            
            edge_sensitivities[(src, dst)] = sensitivity.item()
    
    return edge_sensitivities

def calculate_semantic_sensitivity(model, batch, semantic_flow):
    """计算语义敏感度
    
    Args:
        model: 模型对象
        batch: 批次数据
        semantic_flow: 语义流
        
    Returns:
        semantic_sensitivities: 语义敏感度张量
    """
    # 获取边索引
    edge_index = batch.edge_index
    
    # 初始化语义敏感度
    semantic_sensitivities = torch.zeros(edge_index.shape[1], device=model.device)
    
    # 计算每条边的语义敏感度
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        # 使用语义流的平均值作为敏感度
        semantic_sensitivities[i] = semantic_flow[i].mean()
    
    return semantic_sensitivities

def plot_checkpoint_metrics(history: List[Dict], output_dir: str, k: int = 64):
    """
    绘制训练过程中 Path Success@k、Hits@k、MRR 的趋势曲线，用于 checkpoint 选择
    history: 每 epoch 的验证指标字典，需包含 'path_success', 'hits@k', 'mrr'
    """
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(history) + 1))
    ps = [h.get('path_success', 0) for h in history]
    hs = [h.get(f'hits@{k}', 0) for h in history]
    mrr = [h.get('mrr', 0) for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, ps, label=f'Path Success@{k}')
    plt.plot(epochs, hs, label=f'Hits@{k}')
    plt.plot(epochs, mrr, label='MRR')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Checkpoint Selection Metrics')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'checkpoint_metrics.png'), dpi=300)
    plt.close()

def animate_sensitivity_propagation(graph, sens_list: List[Union[np.ndarray, Dict]], save_path: str, title: str = 'Sensitivity Propagation', interval: int = 500):
    """
    生成敏感度传播动画：sens_list 为一系列边敏感度字典或数组
    graph: PyG 图对象
    save_path: 保存 GIF 路径
    """
    G = to_networkx(graph, to_undirected=False)
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(6, 6))
    edges = list(G.edges())
    def update(frame):
        ax.clear()
        sens = sens_list[frame]
        if isinstance(sens, np.ndarray):
            colors = [sens[i] for i in range(len(edges))]
        else:
            colors = [sens.get(e, 0) for e in edges]
        widths = [1 + 2 * c for c in colors]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, width=widths, edge_cmap=plt.cm.YlOrRd, ax=ax, arrows=True)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax)
        ax.set_title(f"{title} - Step {frame+1}")
        ax.axis('off')
    ani = animation.FuncAnimation(fig, update, frames=len(sens_list), interval=interval)
    ani.save(save_path, writer='imagemagick')
    plt.close()

# 新增：绘制排序和推理相关指标随训练/验证轮数的变化

def plot_ranking_and_reasoning_metrics(history: List[Dict[str, float]],
                                      ks: List[int] = [16, 32, 64],
                                      output_dir: str = './',
                                      title: str = 'Ranking and Reasoning Metrics Over Epochs') -> None:
    """
    绘制 Hits@K、MRR、MAP、Path Success@K 指标随 epoch 变化的曲线

    Args:
        history: 每个 epoch 的指标字典列表，需包含 'hits@K', 'mrr', 'map', 'path_success@K'
        ks: 要绘制的 Top-K 列表
        output_dir: 保存目录
        title: 图表标题
    """
    import os
    epochs = list(range(1, len(history) + 1))
    plt.figure(figsize=(12, 8))
    # Hits@K, MRR, MAP, Path Success@K
    for k in ks:
        hits = [h.get(f'hits@{k}', 0.0) for h in history]
        mrrs = [h.get(f'mrr@{k}', 0.0) for h in history]
        maps = [h.get('map', h.get(f'map@{k}', 0.0)) for h in history]
        ps = [h.get(f'path_success@{k}', 0.0) for h in history]
        plt.plot(epochs, hits, marker='o', label=f'Hits@{k}')
        plt.plot(epochs, mrrs, marker='x', linestyle='--', label=f'MRR@{k}')
        plt.plot(epochs, maps, marker='s', linestyle='-.', label=f'MAP@{k}')
        plt.plot(epochs, ps, marker='^', linestyle=':', label=f'PathSucc@{k}')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'ranking_reasoning_metrics.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"排序和推理指标曲线已保存至: {save_path}")

# 新增：绘制代数一致性指标柱状图

def plot_algebraic_consistency(metrics: Dict[str, float],
                               output_dir: str = './',
                               title: str = 'Algebraic Consistency Metrics') -> None:
    """
    绘制交换律准确率、单位元识别率、逆元识别率等代数一致性指标

    Args:
        metrics: 包含 'exchange_accuracy', 'identity_accuracy', 'inverse_accuracy' 的指标字典
        output_dir: 保存目录
        title: 图表标题
    """
    import os
    labels = []
    values = []
    for key in ['exchange_accuracy', 'identity_accuracy', 'inverse_accuracy']:
        if key in metrics:
            labels.append(key.replace('_', ' ').title())
            values.append(metrics[key])
    plt.figure(figsize=(8, 6))
    sns.barplot(x=labels, y=values, palette='viridis')
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Metric')
    for idx, val in enumerate(values):
        plt.text(idx, val + 0.02, f'{val:.2f}', ha='center')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'algebraic_consistency_metrics.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"代数一致性指标柱状图已保存至: {save_path}")

# 新增：绘制校准质量曲线 (ECE, ACE)

def plot_calibration_metrics(calib: Dict[str, float],
                             output_dir: str = './',
                             title: str = 'Calibration Error Metrics') -> None:
    """
    绘制 ECE 和 ACE 校准误差指标

    Args:
        calib: 包含 'ECE' 与 'ACE' 的字典
        output_dir: 保存目录
        title: 图表标题
    """
    import os
    labels = []
    values = []
    for key in ['ECE', 'ACE']:
        if key in calib:
            labels.append(key)
            values.append(calib[key])
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, palette='magma')
    plt.ylim(0, max(values) + 0.05)
    plt.title(title)
    plt.ylabel('Error')
    plt.xlabel('Metric')
    for idx, val in enumerate(values):
        plt.text(idx, val + 0.005, f'{val:.3f}', ha='center')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'calibration_error_metrics.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"校准误差指标图已保存至: {save_path}")

def generate_evaluation_report(input_path, output_dir, formats=None):
    """
    从评估报告JSON生成可视化
    
    Args:
        input_path: 评估报告JSON文件路径
        output_dir: 输出目录
        formats: 输出格式列表，如 ['html', 'png']
    """
    if formats is None:
        formats = ['html', 'png']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载评估报告
        with open(input_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # 提取路线比较数据
        comparison_data = report_data.get('comparison', {})
        table_data = comparison_data.get('table', [])
        best_accuracy_route = comparison_data.get('best_accuracy_route')
        best_f1_route = comparison_data.get('best_f1_route')
        
        # 提取详细指标数据
        detailed_metrics = report_data.get('detailed_metrics', {})
        
        # 1. 生成准确率比较图
        if table_data:
            # 提取路线和数值
            routes = [row[0] for row in table_data[1:]]  # 跳过表头
            accuracy = [float(row[1]) for row in table_data[1:]]  # 准确率
            precision = [float(row[2]) for row in table_data[1:]]  # 精确率
            recall = [float(row[3]) for row in table_data[1:]]  # 召回率
            f1_scores = [float(row[4]) for row in table_data[1:]]  # F1分数
            inference_times = [float(row[5]) for row in table_data[1:]]  # 推理时间
            
            # 绘制性能比较图
            plt.figure(figsize=(14, 8))
            plt.subplot(2, 2, 1)
            plt.bar(routes, accuracy, color='lightblue')
            plt.ylim(0, 1)
            plt.title('Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.xlabel('Route')
            
            plt.subplot(2, 2, 2)
            plt.bar(routes, f1_scores, color='lightgreen')
            plt.ylim(0, 1)
            plt.title('F1 Score Comparison')
            plt.ylabel('F1 Score')
            plt.xlabel('Route')
            
            plt.subplot(2, 2, 3)
            plt.bar(routes, precision, color='coral', alpha=0.7, label='Precision')
            plt.bar(routes, recall, color='skyblue', alpha=0.7, label='Recall')
            plt.ylim(0, 1)
            plt.title('Precision and Recall')
            plt.ylabel('Score')
            plt.xlabel('Route')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.bar(routes, inference_times, color='lightseagreen')
            plt.title('Average Inference Time (sec)')
            plt.ylabel('Time (sec)')
            plt.xlabel('Route')
            
            plt.tight_layout()
            
            # 保存图像
            if 'png' in formats:
                comparison_path = os.path.join(output_dir, 'performance_comparison.png')
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                print(f"性能比较图已保存至: {comparison_path}")
            
            plt.close()
            
            # 2. 生成雷达图进行综合比较
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Speed']
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # 角度划分
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            # 调整速度评分 (越快越好)
            max_time = max(inference_times)
            speed_scores = [1 - (t / max_time) if max_time > 0 else 0.5 for t in inference_times]
            
            # 绘制每条路线的雷达图
            for i, route in enumerate(routes):
                values = [accuracy[i], precision[i], recall[i], f1_scores[i], speed_scores[i]]
                values += values[:1]  # 闭合图形
                ax.plot(angles, values, linewidth=2, label=f'Route {route}')
                ax.fill(angles, values, alpha=0.1)
            
            # 设置类别标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # 设置Y轴范围
            ax.set_ylim(0, 1)
            
            # 添加图例
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Route Comprehensive Performance Comparison')
            
            # 保存图像
            if 'png' in formats:
                radar_path = os.path.join(output_dir, 'radar_comparison.png')
                plt.savefig(radar_path, dpi=300, bbox_inches='tight')
                print(f"综合性能雷达图已保存至: {radar_path}")
            
            plt.close()
        
        # 3. 生成HTML报告
        if 'html' in formats:
            try:
                # 尝试导入plotly
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots
                
                # 创建性能对比子图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Accuracy Comparison', 'F1 Score Comparison', 'Precision and Recall', 'Average Inference Time (sec)')
                )
                
                # 准确率
                fig.add_trace(
                    go.Bar(x=routes, y=accuracy, name='Accuracy', marker_color='lightblue'),
                    row=1, col=1
                )
                
                # F1分数
                fig.add_trace(
                    go.Bar(x=routes, y=f1_scores, name='F1 Score', marker_color='lightgreen'),
                    row=1, col=2
                )
                
                # 精确率和召回率
                fig.add_trace(
                    go.Bar(x=routes, y=precision, name='Precision', marker_color='coral'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Bar(x=routes, y=recall, name='Recall', marker_color='skyblue'),
                    row=2, col=1
                )
                
                # 推理时间
                fig.add_trace(
                    go.Bar(x=routes, y=inference_times, name='Inference Time', marker_color='lightseagreen'),
                    row=2, col=2
                )
                
                # 更新布局
                fig.update_layout(
                    title='HotpotQA Multi-hop Reasoning Route Performance Comparison',
                    height=800,
                    showlegend=True
                )
                
                # 保存为HTML
                performance_html = os.path.join(output_dir, 'performance_comparison.html')
                fig.write_html(performance_html)
                print(f"交互式性能比较图已保存至: {performance_html}")
                
                # 创建雷达图
                radar_fig = go.Figure()
                
                for i, route in enumerate(routes):
                    values = [accuracy[i], precision[i], recall[i], f1_scores[i], speed_scores[i]]
                    radar_fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f'Route {route}'
                    ))
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title='Route Comprehensive Performance Radar Chart',
                    showlegend=True
                )
                
                # 保存为HTML
                radar_html = os.path.join(output_dir, 'radar_comparison.html')
                radar_fig.write_html(radar_html)
                print(f"交互式雷达图已保存至: {radar_html}")
                
                # 4. 生成汇总报告HTML
                with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>HotpotQA路线评估报告</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f7f9fc; }}
                            .container {{ max-width: 1200px; margin: 0 auto; }}
                            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                            .section {{ background-color: white; margin: 20px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e1e1e1; }}
                            th {{ background-color: #f5f5f5; }}
                            tr:hover {{ background-color: #f9f9f9; }}
                            .highlight {{ background-color: #e7f4e4; }}
                            .visual-section {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                            .visual-item {{ width: 48%; margin-bottom: 20px; }}
                            iframe {{ width: 100%; height: 500px; border: none; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>HotpotQA多跳推理路线评估报告</h1>
                                <p>生成时间：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                            </div>
                            
                            <div class="section">
                                <h2>性能概述</h2>
                                <p>最佳准确率路线: <strong>Route {best_accuracy_route}</strong></p>
                                <p>最佳F1分数路线: <strong>Route {best_f1_route}</strong></p>
                                
                                <h3>性能比较表</h3>
                                <table>
                                    <tr>
                                        <th>路线</th>
                                        <th>准确率</th>
                                        <th>精确率</th>
                                        <th>召回率</th>
                                        <th>F1分数</th>
                                        <th>推理时间(秒)</th>
                                    </tr>
                    """)
                    
                    # 添加表格数据
                    for row in table_data[1:]:  # 跳过表头
                        highlight = ""
                        if str(row[0]) == str(best_accuracy_route) or str(row[0]) == str(best_f1_route):
                            highlight = "class='highlight'"
                        f.write(f"<tr {highlight}><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td><td>{row[5]}</td></tr>\n")
                    
                    f.write(f"""
                                </table>
                            </div>
                            
                            <div class="section">
                                <h2>可视化比较</h2>
                                <div class="visual-section">
                                    <div class="visual-item">
                                        <h3>性能对比图</h3>
                                        <iframe src="performance_comparison.html"></iframe>
                                    </div>
                                    <div class="visual-item">
                                        <h3>综合性能雷达图</h3>
                                        <iframe src="radar_comparison.html"></iframe>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="section">
                                <h2>详细指标</h2>
                    """)
                    
                    # 添加详细指标
                    for route, metrics in detailed_metrics.items():
                        f.write(f"<h3>Route {route}</h3>\n")
                        f.write("<table>\n<tr><th>指标</th><th>值</th></tr>\n")
                        for metric, value in metrics.items():
                            f.write(f"<tr><td>{metric}</td><td>{value}</td></tr>\n")
                        f.write("</table>\n")
                        
                    f.write(f"""
                            </div>
                        </div>
                    </body>
                    </html>
                    """)
                
                print(f"HTML汇总报告已保存至: {os.path.join(output_dir, 'index.html')}")
            
            except ImportError as e:
                print(f"无法生成交互式HTML报告: {e}")
                print("请安装plotly: pip install plotly")
                
                # 退回到生成静态HTML
                with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
                    f.write(f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <title>HotpotQA路线评估报告</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            table {{ border-collapse: collapse; width: 100%; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; }}
                            tr:nth-child(even) {{ background-color: #f2f2f2; }}
                            th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #4CAF50; color: white; }}
                            img {{ max-width: 100%; height: auto; }}
                        </style>
                    </head>
                    <body>
                        <h1>HotpotQA多跳推理路线评估报告</h1>
                        <p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        
                        <h2>性能比较</h2>
                        <table>
                            <tr>
                                <th>路线</th>
                                <th>准确率</th>
                                <th>精确率</th>
                                <th>召回率</th>
                                <th>F1分数</th>
                                <th>推理时间(秒)</th>
                            </tr>
                    """)
                    
                    # 添加表格数据
                    for row in table_data[1:]:  # 跳过表头
                        f.write(f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td><td>{row[5]}</td></tr>\n")
                    
                    f.write(f"""
                        </table>
                        
                        <h2>性能图表</h2>
                        <img src="performance_comparison.png" alt="性能比较">
                        <img src="radar_comparison.png" alt="雷达图比较">
                    </body>
                    </html>
                    """)
                    
                print(f"静态HTML汇总报告已保存至: {os.path.join(output_dir, 'index.html')}")
        
        return True
    
    except Exception as e:
        print(f"生成可视化时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='生成评估报告可视化')
    parser.add_argument('--input', required=True, help='输入评估报告JSON文件路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--format', default='html,png', help='输出格式，逗号分隔，可选项: html,png')
    
    args = parser.parse_args()
    
    formats = args.format.split(',')
    valid_formats = ['html', 'png']
    formats = [fmt for fmt in formats if fmt in valid_formats]
    
    if not formats:
        formats = valid_formats
    
    success = generate_evaluation_report(args.input, args.output, formats)
    
    if success:
        print(f"可视化生成成功，结果保存在: {args.output}")
    else:
        print("可视化生成失败")
        exit(1)

if __name__ == "__main__":
    main()