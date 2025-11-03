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
    num_layers = int(np.ceil(edges / max_edges_per_layer)) # ceil(total edges/max edges per layer) -- what is the min amount of layers needed.
    
    return num_layers

def modify_k33_with_vias(df_k33):
    """
    Modify the K3,3 adjacency matrix by splitting edges with VIA nodes to create 2 planar graphs.
    """
    # Create a copy of the original dataframe to edit.
    df_modified = df_k33.copy()
    
    # Splitting edge between A1 and B1 -- this is just adding the VIA point where we split an edge and remove it from the first plane.
    # By adding the new VIA1 and VIA2, we can then create planar versions of our graphs.
    new_via_node1 = 'VIA1'
    new_via_node2 = 'VIA2'
    splitnode_1, splitnode_2 = 'A1', 'B1'
    new_index = df_modified.index.tolist() + [new_via_node1, new_via_node2]
    df_modified = df_modified.reindex(index=new_index, columns=new_index, fill_value=0)
    
    if df_modified.loc[splitnode_1, splitnode_2] == 1: 
        df_modified.loc[splitnode_1, splitnode_2] = 0
        df_modified.loc[splitnode_2, splitnode_1] = 0
        
        # adding the new path.
        df_modified.loc[splitnode_1, new_via_node1] = 1
        df_modified.loc[new_via_node1, splitnode_1] = 1
        df_modified.loc[new_via_node2, splitnode_2] = 1
        df_modified.loc[splitnode_2, new_via_node2] = 1
    
    # additional nodes -- just to make it look nice, not functionally required 
    # remember that we only need to remove 1 edge because of our 8 max edges per layer.
    # COMPLETELY ARBITRARY!!!
    new_via_node3 = 'VIA3'
    new_via_node4 = 'VIA4'
    splitnode_3, splitnode_4 = 'A3', 'B2'
    
    new_index = df_modified.index.tolist() + [new_via_node3, new_via_node4]
    df_modified = df_modified.reindex(index=new_index, columns=new_index, fill_value=0)
    
    if df_modified.loc[splitnode_3, splitnode_4] == 1:
        df_modified.loc[splitnode_3, splitnode_4] = 0
        df_modified.loc[splitnode_4, splitnode_3] = 0
        df_modified.loc[splitnode_3, new_via_node3] = 1
        df_modified.loc[new_via_node3, splitnode_3] = 1
        df_modified.loc[splitnode_4, new_via_node4] = 1
        df_modified.loc[splitnode_4, new_via_node4] = 1
    
    return df_modified

def visualize_k33_layers_3d(df_modified, via_csv_path, out_folder="Figures", show_plot=True):
    """
    Visualizing our K3,3 Graph in terms of 2 planar graphs.
    """
    os.makedirs(out_folder, exist_ok=True) # we need to load it and save it to the figures.

    # Build graph from modified adjacency matrix (Layer 1)
    G_layer0 = nx.from_pandas_adjacency(df_modified)
    
    # Load VIA connections from adjacency matrix (Layer 2)
    df_via = pd.read_csv(via_csv_path, index_col=0)
    df_via = df_via.reindex(index=df_via.index.union(df_via.columns), columns=df_via.index.union(df_via.columns), fill_value=0)
    df_via = df_via.loc[df_via.columns, df_via.columns]
    G_via = nx.from_pandas_adjacency(df_via)
    
    # Get all nodes from both layers
    nodes_layer0 = set(G_layer0.nodes())
    nodes_via = set(G_via.nodes())
    pos_layer0 = nx.planar_layout(G_layer0)
     
    # For VIA nodes in Layer 2, position them at the same (x,y) as in Layer 1 -- basically we just align the VIAs on x and y.
    pos_layer1 = {}
    for node in nodes_via:
        if node in pos_layer0:
            pos_layer1[node] = pos_layer0[node]
        else:
            pos_layer1[node] = np.array([0.0, 0.0])
    
    # Creating the actual 3D graph -- initializing points as a 3D graph.
    pos_3d_layer0 = {node: (float(pos_layer0[node][0]), float(pos_layer0[node][1]), 0.0)  for node in nodes_layer0}
    pos_3d_layer1 = {node: (float(pos_layer1[node][0]), float(pos_layer1[node][1]), 1.0)  for node in nodes_via}

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Draw semi-transparent planes with higher opacity -- we set the bounds of our grid and establish our 3D plane.
    all_x = [pos_layer0[n][0] for n in pos_layer0]
    all_y = [pos_layer0[n][1] for n in pos_layer0] 
    # add padding to make the plane larger than the size of the graph -- just makes it look nicer.
    xmin, xmax = min(all_x) - 0.25, max(all_x) + 0.25 
    ymin, ymax = min(all_y) - 0.25, max(all_y) + 0.25 
    plane_layer1 = [(xmin, ymin, 0.0), (xmax, ymin, 0.0), (xmax, ymax, 0.0), (xmin, ymax, 0.0)] # establishing the PCB Layer 1
    plane_layer2 = [(xmin, ymin, 1.0), (xmax, ymin, 1.0), (xmax, ymax, 1.0), (xmin, ymax, 1.0)] # Establishing the PCB Layer 2
    poly_layer1 = Poly3DCollection([plane_layer1], alpha=0.35, facecolor=(0.0, 0.3, 0.3))
    poly_layer2 = Poly3DCollection([plane_layer2], alpha=0.35, facecolor=(0.3, 0.1, 0.0))
    ax.add_collection3d(poly_layer1)
    ax.add_collection3d(poly_layer2)
    
    # Draw high-opacity borders around both layers -- this is just aesthetics, completely arbitrary.
    border_layer1 = [
        [(xmin, ymin, 0.0), (xmax, ymin, 0.0)],
        [(xmax, ymin, 0.0), (xmax, ymax, 0.0)],
        [(xmax, ymax, 0.0), (xmin, ymax, 0.0)],
        [(xmin, ymax, 0.0), (xmin, ymin, 0.0)]
    ]
    border_layer1_finished = Line3DCollection(border_layer1, colors=(0.0, 0.8, 0.8), linewidths=1, alpha=0.6)
    ax.add_collection3d(border_layer1_finished)
    border_layer2_edges = [
        [(xmin, ymin, 1.0), (xmax, ymin, 1.0)],
        [(xmax, ymin, 1.0), (xmax, ymax, 1.0)],
        [(xmax, ymax, 1.0), (xmin, ymax, 1.0)],
        [(xmin, ymax, 1.0), (xmin, ymin, 1.0)]
    ]
    border_layer2 = Line3DCollection(border_layer2_edges, colors=(0.8, 0.3, 0.0), linewidths=1, alpha=0.6)
    ax.add_collection3d(border_layer2)
    
    # Draw edges for Layer 1
    for u, v in G_layer0.edges():
        x = [pos_3d_layer0[u][0], pos_3d_layer0[v][0]]
        y = [pos_3d_layer0[u][1], pos_3d_layer0[v][1]]
        z = [0, 0]
        if (u, v) == list(G_layer0.edges())[0]:
            ax.plot(x, y, z, color='cyan', linewidth=2.5, alpha=0.9, label='Layer 1')
        else:
            ax.plot(x, y, z, color='cyan', linewidth=2.5, alpha=0.9)
    
    # Draw nodes for Layer 1 
    for node in nodes_layer0:
        x, y, z = pos_3d_layer0[node]
        if node.startswith('VIA'):
            color = '#FFD700' 
        else:
            color = 'white'
        ax.scatter(x, y, z, s=100, color=color, edgecolors='black', linewidths=1.5, zorder=10)
        ax.text(x, y, z - 0.08, node, color='white', fontsize=9, ha='center', va='top', weight='bold')
    
    # Draw edges for VIA layer (2nd layer)
    for u, v in G_via.edges():
        x = [pos_3d_layer1[u][0], pos_3d_layer1[v][0]]
        y = [pos_3d_layer1[u][1], pos_3d_layer1[v][1]]
        z = [1, 1]
        if (u, v) == list(G_via.edges())[0]:
            ax.plot(x, y, z, color='orange', linewidth=2.5, alpha=0.9, label='Layer 2')
        else:
            ax.plot(x, y, z, color='orange', linewidth=2.5, alpha=0.9)
    
    # Draw nodes for VIA layer (2nd layer)
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
            if node == list(nodes_via)[0]:
                ax.plot([x, x], [y, y], z, color='yellow', linewidth=2.0, linestyle='--', alpha=0.9, label='Vertical VIA')
            else:
                ax.plot([x, x], [y, y], z, color='yellow', linewidth=2.0, linestyle='--', alpha=0.9)
    
    # Visualization Settings:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(-0.2, 1.4)
    ax.view_init(elev=25, azim=-65)
    
    # Set black panes and remove all visible white lines, ticks, and frames
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticks([])
        axis.line.set_color((0, 0, 0, 0))
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.pane.set_facecolor('black')
        axis.pane.set_alpha(1.0)

    ax.grid(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_axis_off()

    legend = ax.legend(loc='upper left', fontsize=9)
    legend.set_frame_on(False)

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
