import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
import os

def count_necessary_layers(df_full):
    """
    Extracts the K3,3 subgraph, computes the minimum amount of layers needed, finds a planar configuration of our layers.
    """
    # Isolate the k3,3 in our CSV -- we need to do this to compute an accurate amount of layers we need -- denoted by eulers formula.
    k3_3_nodes = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    G_full = nx.from_pandas_adjacency(df_full)
    G_k3_3 = G_full.subgraph(k3_3_nodes)
    
    # finding the minimum amount of layers with eulers formula relationship. 
    verticies = G_k3_3.number_of_nodes()   # 6 vertices
    edges = G_k3_3.number_of_edges()   # 9 edges
    max_edges_per_layer = 2 * verticies - 4  # Euler planar limit: m less than or equal to 2v - 4
    num_layers = int(np.ceil(edges / max_edges_per_layer)) # ceil(total edges/max edges per layer)
    
    return num_layers

def modify_k33_with_vias(df_k33):
    """
    Modify the K3,3 adjacency matrix by splitting the A1-B1 edge with VIA nodes
    (minimal changes: adds VIA1/VIA2 and VIA3/VIA4 as in earlier agreement).
    """
    # Create a copy of the original dataframe
    df_modified = df_k33.copy()
    
    # --- existing VIA pair 1: split A1--B1 ---
    new_via_node1 = 'VIA1'
    new_via_node2 = 'VIA2'
    node_u, node_v = 'A1', 'B1'  # the edge we are splitting
    
    new_index = df_modified.index.tolist() + [new_via_node1, new_via_node2]
    df_modified = df_modified.reindex(index=new_index, columns=new_index, fill_value=0)
    
    if df_modified.loc[node_u, node_v] == 1: 
        df_modified.loc[node_u, node_v] = 0
        df_modified.loc[node_v, node_u] = 0
        
        # Add the new path: A1 -- VIA1  and VIA2 -- B1
        df_modified.loc[node_u, new_via_node1] = 1
        df_modified.loc[new_via_node1, node_u] = 1
        df_modified.loc[new_via_node2, node_v] = 1
        df_modified.loc[node_v, new_via_node2] = 1
    
    # additional nodes -- just to make it look nice, not functionally required.
    new_via_node3 = 'VIA3'
    new_via_node4 = 'VIA4'
    node_u2, node_v2 = 'A3', 'B2'
    
    new_index = df_modified.index.tolist() + [new_via_node3, new_via_node4]
    df_modified = df_modified.reindex(index=new_index, columns=new_index, fill_value=0)
    
    if df_modified.loc[node_u2, node_v2] == 1:
        df_modified.loc[node_u2, node_v2] = 0
        df_modified.loc[node_v2, node_u2] = 0
        df_modified.loc[node_u2, new_via_node3] = 1
        df_modified.loc[new_via_node3, node_u2] = 1
        df_modified.loc[node_v2, new_via_node4] = 1
        df_modified.loc[node_v2, new_via_node4] = 1
    
    return df_modified

def visualize_k33_layers_3d(df_modified, via_csv_path, out_folder="Figures", show_plot=True):
    """
    Visualizes the K3,3 circuit in 3D with two distinct planar layers:
    - Layer 0 (z=0, bottom): Modified planar graph with pad-to-VIA connections
    - Layer 1 (z=1, top): VIA-to-VIA connections only
    VIA pairs are vertically aligned.
    """
    os.makedirs(out_folder, exist_ok=True)

    # Build graph from modified adjacency matrix (Layer 0)
    G_layer0 = nx.from_pandas_adjacency(df_modified)
    
    # Load VIA connections from adjacency matrix (Layer 1)
    df_via = pd.read_csv(via_csv_path, index_col=0)
    df_via = df_via.reindex(index=df_via.index.union(df_via.columns),
                            columns=df_via.index.union(df_via.columns),
                            fill_value=0)
    df_via = df_via.loc[df_via.columns, df_via.columns]
    G_via = nx.from_pandas_adjacency(df_via)
    
    # Check planarity
    is_planar_0, _ = nx.check_planarity(G_layer0)
    is_planar_via, _ = nx.check_planarity(G_via)
    print(f"Planarity Check → Layer 0 (modified): {is_planar_0}, VIA Layer: {is_planar_via}")
    
    # Get all nodes from both layers
    nodes_layer0 = set(G_layer0.nodes())
    nodes_via = set(G_via.nodes())
    
    # Create layout for Layer 0 (use planar layout if possible)
    try:
        if is_planar_0:
            pos_layer0 = nx.planar_layout(G_layer0)
        else:
            pos_layer0 = nx.spring_layout(G_layer0, seed=42, k=1.5, iterations=50)
    except:
        pos_layer0 = nx.spring_layout(G_layer0, seed=42, k=1.5, iterations=50)
    
    # For VIA nodes in layer 1, position them at the same (x,y) as in layer 0
    # This ensures vertical alignment
    pos_layer1 = {}
    for node in nodes_via:
        if node in pos_layer0:
            pos_layer1[node] = pos_layer0[node]
        else:
            pos_layer1[node] = np.array([0.0, 0.0])
    
    # Convert to 3D coordinates
    pos_3d_layer0 = {node: (float(pos_layer0[node][0]), float(pos_layer0[node][1]), 0.0) 
                     for node in nodes_layer0}
    pos_3d_layer1 = {node: (float(pos_layer1[node][0]), float(pos_layer1[node][1]), 1.0) 
                     for node in nodes_via}
    
    # Create 3D plot with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_title("3D Planar K3,3 Circuit: Two-Layer PCB with VIAs", 
                 color='white', fontsize=14, pad=20)
    
    # Draw semi-transparent planes with higher opacity
    all_x = [pos_layer0[n][0] for n in pos_layer0]
    all_y = [pos_layer0[n][1] for n in pos_layer0]
    xmin, xmax = min(all_x) - 0.3, max(all_x) + 0.3
    ymin, ymax = min(all_y) - 0.3, max(all_y) + 0.3
    
    plane_z0 = [(xmin, ymin, 0.0), (xmax, ymin, 0.0), (xmax, ymax, 0.0), (xmin, ymax, 0.0)]
    plane_z1 = [(xmin, ymin, 1.0), (xmax, ymin, 1.0), (xmax, ymax, 1.0), (xmin, ymax, 1.0)]
    
    poly0 = Poly3DCollection([plane_z0], alpha=0.35, facecolor=(0.0, 0.3, 0.3))
    poly1 = Poly3DCollection([plane_z1], alpha=0.35, facecolor=(0.3, 0.1, 0.0))
    ax.add_collection3d(poly0)
    ax.add_collection3d(poly1)
    
    # Draw high-opacity borders around each layer
    # Layer 0 border (cyan/teal color)
    border_z0_edges = [
        [(xmin, ymin, 0.0), (xmax, ymin, 0.0)],
        [(xmax, ymin, 0.0), (xmax, ymax, 0.0)],
        [(xmax, ymax, 0.0), (xmin, ymax, 0.0)],
        [(xmin, ymax, 0.0), (xmin, ymin, 0.0)]
    ]
    border_z0 = Line3DCollection(border_z0_edges, colors=(0.0, 0.8, 0.8), linewidths=3, alpha=0.6)
    ax.add_collection3d(border_z0)
    
    # Layer 1 border (orange/red color)
    border_z1_edges = [
        [(xmin, ymin, 1.0), (xmax, ymin, 1.0)],
        [(xmax, ymin, 1.0), (xmax, ymax, 1.0)],
        [(xmax, ymax, 1.0), (xmin, ymax, 1.0)],
        [(xmin, ymax, 1.0), (xmin, ymin, 1.0)]
    ]
    border_z1 = Line3DCollection(border_z1_edges, colors=(0.8, 0.3, 0.0), linewidths=1, alpha=0.6)
    ax.add_collection3d(border_z1)
    
    # Draw edges for Layer 0 (modified graph with pad-VIA connections)
    for u, v in G_layer0.edges():
        x = [pos_3d_layer0[u][0], pos_3d_layer0[v][0]]
        y = [pos_3d_layer0[u][1], pos_3d_layer0[v][1]]
        z = [0, 0]
        ax.plot(x, y, z, color='cyan', linewidth=2.5, alpha=0.9, label='Layer 0' if (u,v) == list(G_layer0.edges())[0] else '')
    
    # Draw nodes for Layer 0
    for node in nodes_layer0:
        x, y, z = pos_3d_layer0[node]
        color = '#FFD700' if node.startswith('VIA') else 'white'  # Gold for VIAs
        ax.scatter(x, y, z, s=100, color=color, edgecolors='black', linewidths=1.5, zorder=10)
        ax.text(x, y, z - 0.08, node, color='white', fontsize=9, ha='center', va='top', weight='bold')
    
    # Draw edges for VIA layer (VIA-to-VIA connections)
    for u, v in G_via.edges():
        x = [pos_3d_layer1[u][0], pos_3d_layer1[v][0]]
        y = [pos_3d_layer1[u][1], pos_3d_layer1[v][1]]
        z = [1, 1]
        ax.plot(x, y, z, color='orange', linewidth=2.5, alpha=0.9, label='VIA Layer' if (u,v) == list(G_via.edges())[0] else '')
    
    # Draw nodes for VIA layer
    for node in nodes_via:
        x, y, z = pos_3d_layer1[node]
        ax.scatter(x, y, z, s=100, color='#FFD700', edgecolors='black', linewidths=1.5, zorder=10)
        ax.text(x, y, z + 0.08, node, color='white', fontsize=9, ha='center', va='bottom', weight='bold')
    
    # Draw vertical lines connecting VIA nodes between layers
    for node in nodes_via:
        if node in pos_3d_layer0:  # VIA exists in both layers
            x = pos_3d_layer0[node][0]
            y = pos_3d_layer0[node][1]
            z = [0, 1]
            ax.plot([x, x], [y, y], z, color='yellow', linewidth=2.0, 
                   linestyle='--', alpha=0.9, label='Vertical VIA' if node == list(nodes_via)[0] else '')
    
    # Visualization Settings:

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-0.2, 1.4)
    ax.set_xlabel("X", color='white', fontsize=11)
    ax.set_ylabel("Y", color='white', fontsize=11)
    ax.set_zlabel("Layer (Z)", color='white', fontsize=11)
    ax.view_init(elev=25, azim=-65)
    ax.xaxis.pane.set_facecolor('black')
    ax.yaxis.pane.set_facecolor('black')
    ax.zaxis.pane.set_facecolor('black')
    ax.xaxis.pane.set_alpha(1.0)
    ax.yaxis.pane.set_alpha(1.0)
    ax.zaxis.pane.set_alpha(1.0)
    ax.grid(True, color='white', alpha=0.3, linestyle='-', linewidth=0.3)
    ax.xaxis.line.set_color('white')
    ax.yaxis.line.set_color('white')
    ax.zaxis.line.set_color('white')
    ax.tick_params(colors='white', labelsize=9)
    ax.legend(loc='upper left', fontsize=9)
    
    # Saving the static image and the animation:
    static_path = os.path.join(out_folder, "Planar K3,3 3D (static).png")
    plt.tight_layout()
    plt.savefig(static_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    num_frames = 360 

    def animate(frame):
        """Rotate the view 360 degrees around the z-axis"""
        angle = frame * 1
        ax.view_init(elev=25, azim=-65 + angle, vertical_axis='z')
        return ax,
    
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=True, blit=False)
    gif_path = os.path.join(out_folder, "Planar K3_3 3D (animation).gif")
    anim.save(gif_path, writer='pillow', fps=30)
    print(f"Saved 360° rotation animation to: {gif_path}")

    return {
        "edges_layer0": list(G_layer0.edges()),
        "edges_via": list(G_via.edges()),
        "static_path": static_path,
        "gif_path": gif_path
    }

# main function to run the code:
if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "CSV")
    df_full = pd.read_csv(os.path.join(base, "Nonplanar_K3,3.csv"), index_col=0)
    layers_needed = count_necessary_layers(df_full)
    print(f"Layers needed based on Euler's formula: {layers_needed}")

    df_mod = modify_k33_with_vias(df_full)
    via_csv = os.path.join(base, "VIA.csv")
    result = visualize_k33_layers_3d(df_mod, via_csv, out_folder="Figures", show_plot=True)