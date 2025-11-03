import networkx as nx
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import sys
import matplotlib.patheffects as path_effects

def check_planarity(csv_path):
    """
    Load the CSV and check if the graph is planar using NetworkX.
    Returns the graph object and planarity status.
    """
    # Load the non-planar K3,3 graph
    dataframe = pd.read_csv(csv_path, index_col=0)
    # Ensure the index matches the columns for NetworkX
    dataframe.index = dataframe.columns
    K33_graph = nx.from_pandas_adjacency(dataframe)

    print(f"Graph with {K33_graph.number_of_nodes()} nodes and {K33_graph.number_of_edges()} edges")
    print("Nodes:", list(K33_graph.nodes()))
    print("Edges:", list(K33_graph.edges()))

    # Check Planarity
    is_planar = nx.is_planar(K33_graph)
    print(f"K3,3 Graph is planar: {is_planar}")
    
    return K33_graph, is_planar

def graph_nonplanar(K33_graph):
    """
    Create the animated visualization showing the algorithm searching for a planar layout.
    Returns the figure, animation, and frame parameters.
    """
    # Generate multiple random layouts for the oscillation
    NUM_LAYOUTS = 6
    np.random.seed(42)  # For reproducibility
    pos_layouts = [nx.spring_layout(K33_graph, k=1.5, iterations=50, seed=i*100) for i in range(NUM_LAYOUTS)]

    # Get node colors and labels - Assign color based on K3,3 bipartite structure:
    # A nodes (A1, A2, A3) in blue, B nodes (B1, B2, B3) in red
    # A nodes represent varying power supplies (3.3, 5, 12 volts for example).
    node_colors = []
    for node in K33_graph.nodes():
        if node.startswith('A'):
            node_colors.append('#00BFFF')  # Blue for A nodes
        elif node.startswith('B'):
            node_colors.append('#FF1744')  # Red for B nodes
        else:
            node_colors.append('#FFFFFF')  # White default
    
    # Make the labels for the dots in our graph just the pin number associated.
    labels = {}
    for node in K33_graph.nodes():
        if node == '5V':
            labels[node] = '5V'
        elif node == 'GND':
            labels[node] = 'GND'
        else:
            parts = node.split('_')
            if len(parts) > 1:
                labels[node] = parts[-1]
            else:
                labels[node] = node

    # Animation Control Variables
    MOVEMENT_FRAMES = 100 
    PAUSE_FRAMES = 40
    num_frames = MOVEMENT_FRAMES + PAUSE_FRAMES

    # Create the figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis('off')

    # Build legend for K3,3 bipartite graph
    legend_elements = [
        Patch(facecolor='#1F78B4', label='A Nodes (power supply nodes)'),
        Patch(facecolor='#E31A1C', label='B Nodes (electrical component nodes)')
    ]

    def animate(frame):
        """
        Animation function that shows the algorithm searching for a planar layout.
        """
        # Reset the frame
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')
        
        # Calculate time parameter
        if frame < MOVEMENT_FRAMES:
            t = frame / (MOVEMENT_FRAMES - 1)
        else:
            # Pause at the last oscillation position
            t = 1.0
        
        # Oscillate between multiple random layouts using a sine wave pattern.
        # This simulates the algorithm "searching" for a planar layout but never settling.
        # Create a sine-based oscillation that speeds up and slows down
        # Frequency controls how many times we oscillate through all layouts
        frequency = 0.5  # Complete cycles through all layouts
        
        # Sine wave that oscillates between 0 and num_layouts-1
        # The sine wave naturally speeds up and slows down, similar to our easing function
        layout_index_float = (np.sin(2 * np.pi * frequency * t) + 1) / 2 * (NUM_LAYOUTS - 1)
        
        # Get the two layouts we're interpolating between
        layout_index_low = int(np.floor(layout_index_float))
        layout_index_high = int(np.ceil(layout_index_float))
        
        # Handle edge case where we're exactly at the last layout
        if layout_index_high >= NUM_LAYOUTS:
            layout_index_high = NUM_LAYOUTS - 1
        
        # Interpolation factor between the two layouts
        t_between = layout_index_float - layout_index_low
        
        # Apply easing to the interpolation for smoother transitions
        # Use a cubic easing function similar to the original
        if t_between < 0.5:
            t_eased = 2 * t_between * t_between
        else:
            t_eased = 1 - 2 * (1 - t_between) * (1 - t_between)
        
        # Interpolate between the two layouts
        pos_current = {}
        pos_low = pos_layouts[layout_index_low]
        pos_high = pos_layouts[layout_index_high]
        
        for node in pos_low:
            x1, y1 = pos_low[node]
            x2, y2 = pos_high[node]
            pos_current[node] = (
                x1 + (x2 - x1) * t_eased,
                y1 + (y2 - y1) * t_eased
            )
        
        # Draw the graph
        nx.draw_networkx_edges(K33_graph, pos_current, ax=ax, edge_color='white', width=1.5, alpha=0.7)
        nx.draw_networkx_nodes(K33_graph, pos_current, ax=ax, node_color=node_colors, node_size=300)
        
        for node, (x, y) in pos_current.items():
            text = ax.text(x, y, labels[node], fontsize=10, fontweight='bold', 
                        ha='center', va='center', color='white')
            text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])  

        # Add legend
        legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        legend.set_frame_on(False)

    # Create animation
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=True)
    
    return fig, anim, animate, MOVEMENT_FRAMES

def save_visualization(fig, anim, animate_func, movement_frames, output_dir="Figures"):
    """
    Save the visualization as both a GIF animation and a static PNG image.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save animation as GIF
    output_path = os.path.join(output_dir, "Nonplanar K3,3 (animation).gif")
    anim.save(output_path, writer='pillow', fps=30)
    print(f"Saved animation: {output_path}")

    # Save static image of a representative frame (middle of animation)
    animate_func(movement_frames // 2)
    static_image_path = os.path.join(output_dir, "Nonplanar K3,3 (static).png")
    plt.savefig(static_image_path, facecolor=fig.get_facecolor())
    print(f"Saved static image: {static_image_path}")

# main function to run the code:
if __name__ == "__main__":
    csv_path = os.path.join(os.getcwd(), "CSV", "Nonplanar_K3,3.csv")
    K33_graph, is_planar = check_planarity(csv_path)
    fig, anim, animate_func, movement_frames = graph_nonplanar(K33_graph)
    save_visualization(fig, anim, animate_func, movement_frames, output_dir="Figures")
    
    plt.show()