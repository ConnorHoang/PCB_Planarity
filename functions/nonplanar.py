"""
This file checks the planarity of a graph. When the graph is non-planar,
the minimum number of nodes nedded to make the graph planr will  be found.

The code will also visualize the graph and the process of making it planar.
"""

import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import sys


def visualize_adjacency_matrix(csv_path, output_dir=None):
    """
    Visualize the adjacency matrix as a graph with nodes and edges.
    
    Args:
        csv_path: Path to the CSV file containing the adjacency matrix
        output_dir: Directory to save the visualization (default: circuit_diagrams in project root)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)
    
    # Convert to numpy array
    adjacency_matrix = df.values
    
    # Create NetworkX graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Relabel nodes to match CSV column names (if they exist)
    node_labels = df.columns.tolist()
    if node_labels:
        mapping = {i: str(node_labels[i]) for i in range(len(node_labels))}
        G = nx.relabel_nodes(G, mapping)
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for a good looking graph
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=1500,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    nx.draw_networkx_edges(G, pos,
                          width=2,
                          alpha=0.6,
                          edge_color='gray',
                          style='solid')
    
    nx.draw_networkx_labels(G, pos,
                           font_size=14,
                           font_weight='bold')
    
    plt.title('Graph Visualization from Adjacency Matrix', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    if output_dir is None:
        # Default to project_root/circuit_diagrams
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'circuit_diagrams')
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'adjacency_matrix_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close()
    
    return adjacency_matrix, df


if __name__ == "__main__":
    # Visualize the non_planar graph
    # Get the directory of this script and construct path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "circuits", "non_planar - Sheet1.csv")
    adjacency_matrix, df = visualize_adjacency_matrix(csv_path)
    

