"""
A file that visualizes non-planar graphs attempting (and failing) to find a planar layout.
The animation shows the algorithm searching through different layouts.
"""

import networkx as nx
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import sys

# Dark Theme Colors for Graph
QB_DARK_BG = '#121212'
QB_DARK_TEXT = '#FFFFFF'
QB_COLORS = ['#E31A1C', '#33A02C', '#FF7F00', '#1F78B4', '#6A3D9A',
             '#FB9A99', '#B2DF8A', '#FDBF6F', '#A6CEE3', '#CAB2D6']

def get_node_color(node_name):
    """
    Assign color based on K3,3 bipartite structure:
    A nodes (A1, A2, A3) in blue
    B nodes (B1, B2, B3) in red
    """
    if node_name.startswith('A'):
        return '#1F78B4'  # Blue for A nodes
    elif node_name.startswith('B'):
        return '#E31A1C'  # Red for B nodes
    else:
        return '#FFFFFF'  # White default

def get_short_label(node_name):
    """
    Keep full labels for K3,3 nodes (A1, A2, A3, B1, B2, B3)
    """
    return node_name

def get_short_label(node_name):
    """
    Make the labels for the dots in our graph just the pin number associated.
    """
    if node_name == '5V': 
        return '5V'
    elif node_name == 'GND':
        return 'GND'
    else:
        parts = node_name.split('_')
        if len(parts) > 1:
            return parts[-1]
        return node_name


def oscillate_positions(pos_layouts, t, num_layouts=5):
    """
    Oscillate between multiple random layouts using a sine wave pattern.
    This simulates the algorithm "searching" for a planar layout but never settling.
    
    pos_layouts: list of position dictionaries (multiple random layouts)
    t: time parameter from 0.0 to 1.0
    num_layouts: number of layouts to oscillate between
    """
    # Create a sine-based oscillation that speeds up and slows down
    # Frequency controls how many times we oscillate through all layouts
    frequency = 0.5  # Complete cycles through all layouts
    
    # Sine wave that oscillates between 0 and num_layouts-1
    # The sine wave naturally speeds up and slows down, similar to our easing function
    layout_index_float = (np.sin(2 * np.pi * frequency * t) + 1) / 2 * (num_layouts - 1)
    
    # Get the two layouts we're interpolating between
    layout_index_low = int(np.floor(layout_index_float))
    layout_index_high = int(np.ceil(layout_index_float))
    
    # Handle edge case where we're exactly at the last layout
    if layout_index_high >= num_layouts:
        layout_index_high = num_layouts - 1
    
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
    
    return pos_current


def build_dynamic_legend(graph):
    """
    Build legend for K3,3 bipartite graph
    """
    legend_elements = [
        Patch(facecolor='#1F78B4', label='A Nodes (A1, A2, A3)'),
        Patch(facecolor='#E31A1C', label='B Nodes (B1, B2, B3)')
    ]
    return legend_elements


# Load the non-planar K3,3 graph
dataframe = pd.read_csv('CSV/Nonplanar_K3,3.csv', index_col=0)
# Ensure the index matches the columns for NetworkX
dataframe.index = dataframe.columns
K33_graph = nx.from_pandas_adjacency(dataframe)

print(f"Graph with {K33_graph.number_of_nodes()} nodes and {K33_graph.number_of_edges()} edges")
print("Nodes:", list(K33_graph.nodes()))
print("Edges:", list(K33_graph.edges()))

# Check Planarity
is_planar = nx.is_planar(K33_graph)
print(f"K3,3 Graph is planar: {is_planar}")

# Generate multiple random layouts for the oscillation
NUM_LAYOUTS = 6
np.random.seed(42)  # For reproducibility
pos_layouts = [nx.spring_layout(K33_graph, k=1.5, iterations=50, seed=i*100) for i in range(NUM_LAYOUTS)]

# Get node colors and labels
node_colors = [get_node_color(node) for node in K33_graph.nodes()]
labels = {node: get_short_label(node) for node in K33_graph.nodes()}

# Animation Control Variables
MOVEMENT_FRAMES = 100  # Match the planar circuit animation length
PAUSE_FRAMES = 40
num_frames = MOVEMENT_FRAMES + PAUSE_FRAMES

# Create output directory
output_dir = os.path.join(os.getcwd(), "Figures")
os.makedirs(output_dir, exist_ok=True)

# Create the figure
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(QB_DARK_BG)
ax.set_facecolor(QB_DARK_BG)
ax.axis('on')  # Turn on axis to show grid
ax.grid(True, color=QB_DARK_TEXT, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_xticks([])  # Hide tick marks
ax.set_yticks([])  # Hide tick marks
for spine in ax.spines.values():
    spine.set_visible(False)  # Hide border

# Build legend
legend_elements = build_dynamic_legend(K33_graph)


def animate(frame):
    """
    Animation function that shows the algorithm searching for a planar layout.
    """
    # Reset the frame
    ax.clear()
    ax.set_facecolor(QB_DARK_BG)
    ax.axis('on')  # Turn on axis to show grid
    ax.grid(True, color=QB_DARK_TEXT, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_xticks([])  # Hide tick marks
    ax.set_yticks([])  # Hide tick marks
    for spine in ax.spines.values():
        spine.set_visible(False)  # Hide border
    
    # Calculate time parameter
    if frame < MOVEMENT_FRAMES:
        t = frame / (MOVEMENT_FRAMES - 1)
    else:
        # Pause at the last oscillation position
        t = 1.0
    
    # Get current positions using oscillation
    pos_current = oscillate_positions(pos_layouts, t, NUM_LAYOUTS)
    
    # Draw the graph
    nx.draw_networkx_edges(K33_graph, pos_current, ax=ax, 
                          edge_color=QB_DARK_TEXT, width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(K33_graph, pos_current, ax=ax,
                          node_color=node_colors, node_size=300)
    nx.draw_networkx_labels(K33_graph, pos_current, labels, ax=ax,
                           font_color='#000000', font_size=10, 
                           font_weight='bold')
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
             facecolor=QB_DARK_BG, edgecolor=QB_DARK_TEXT, labelcolor=QB_DARK_TEXT)


# Create animation
anim = FuncAnimation(fig, animate, frames=num_frames, interval=50, repeat=True)

# Save animation as GIF
output_path = os.path.join(output_dir, "Nonplanar K3,3 (animation).gif")
anim.save(output_path, writer='pillow', fps=30)
print(f"Saved animation: {output_path}")

# Save static image of a representative frame (middle of animation)
animate(MOVEMENT_FRAMES // 2)
static_image_path = os.path.join(output_dir, "Nonplanar K3,3 (static).png")
plt.savefig(static_image_path, facecolor=fig.get_facecolor())
print(f"Saved static image: {static_image_path}")

plt.show()