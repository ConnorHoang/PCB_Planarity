"""
A file that tests planarity of our planar graphs (LED Circuit and Massive Circuits). 
Afterwards, we save the output as a GIF and PNG.
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

# Dark Theme Colors for Graph -- we need to find something that might be better?
QB_DARK_BG = '#121212'
QB_DARK_TEXT = '#FFFFFF'
QB_COLORS = ['#E31A1C', '#33A02C', '#FF7F00', '#1F78B4', '#6A3D9A',
             '#FB9A99', '#B2DF8A', '#FDBF6F', '#A6CEE3', '#CAB2D6']

# Component color mapping -- we just label the respective nodes the numbers associated.
COMPONENT_COLORS = {
    '5V': '#E31A1C',           # Red
    'GND': '#E31A1C',          # Also red.
    'Switch': '#1F78B4',       # Blue 
    'LED': '#FF7F00',          # Orange 
    'Resistor': '#FDBF6F',     # Light orange
    'Capacitor': '#6A3D9A',    # Purple 
    'Transistor': '#33A02C',   # Green
    'Buzzer': '#CAB2D6',       # Light purple
}

# Map the color to the respective nodes.
def get_node_color(node_name):
    """
    assign the respective node to the respective color.
    """
    if node_name == '5V':
        return COMPONENT_COLORS['5V']
    elif node_name == 'GND':
        return COMPONENT_COLORS['GND']
    elif 'Switch' in node_name:
        return COMPONENT_COLORS['Switch']
    elif 'LED' in node_name:
        return COMPONENT_COLORS['LED']
    elif 'Resistor' in node_name:
        return COMPONENT_COLORS['Resistor']
    elif 'Capacitor' in node_name:
        return COMPONENT_COLORS['Capacitor']
    elif 'Transistor' in node_name:
        return COMPONENT_COLORS['Transistor']
    elif 'Buzzer' in node_name:
        return COMPONENT_COLORS['Buzzer']
    else:
        return '#FFFFFF'    

# Changing labels of our dots -- 5v and GND do not matter, we only care about the other labels.
# The labeling is pretty arbitrary, but we do use numbers to label the nodes for the respective pins on the components.
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

# this is a function that utilizes a cubic relationship to ease in and out of our animations.
# it takes in pos1 and pos2, our starting and ending graphs--'nonplanar' representation v.s. our planar rep.
# we also represent time as a fraction of 1 -- floats representing percentage through the graph.
def interpolate_positions(pos1, pos2, t):
    """
    Ease in and our of our animation using interpolation.
    """
    # C controls the steepness / intensity of animation acceleration/deceleration.
    C = 6.0

    # Load the adjacency matrix - LED circuit (1)
    if t == 1.0:
        # we have finished our graph -- we have the planar version.
        t_eased = 1.0
    elif t < 0.5:
        # starts slow, gets fast -- first half of our animation.
        # Formula: 0.5 * 2^(C * (2t - 1))
        t_eased = 0.5 * np.power(2.0, C * (2.0 * t - 1.0))
    else:
        # starts fast, gets very slow near 1.0 - this is the latter half of our animation.
        # Formula: 0.5 * (2 - 2^(-C * (2t - 1))) -- closest we can get to negative exponential relationship.
        # The use of 2t - 1 term centers the function, guaranteeing a smooth meeting point at t=0.5.
        t_eased = 0.5 * (2.0 - np.power(2.0, -C * (2.0 * t - 1.0)))

    # Applying interpolation to get us to our final graph.
    pos_interp = {}
    # for every node in the graph.
    for node in pos1:
        x1, y1 = pos1[node] # start pos.
        x2, y2 = pos2[node] # end pos.
        pos_interp[node] = (
            # Interpolate X coordinate
            x1 + (x2 - x1) * t_eased,
            # Interpolate Y coordinate
            y1 + (y2 - y1) * t_eased
        )
    return pos_interp

# legend that adjusts for the right graph we are visualizing.
def build_dynamic_legend(graph):
    """
    Build a dynamic legend based on the components actually present in the given graph.
    Only includes components that appear in the node names.
    """
    node_names = [str(n) for n in graph.nodes()]
    used_components = set()
    for node in node_names:
        for comp_name in COMPONENT_COLORS:
            if comp_name != 'GND' and comp_name != '5V' and comp_name.lower() in node.lower():
                used_components.add(comp_name)
        if node == '5V':
            used_components.add('5V')
        if node == 'GND':
            used_components.add('GND')
    legend_elements = [
        Patch(facecolor=COMPONENT_COLORS[c], label=c) for c in used_components
    ]
    return legend_elements

# Load the adjacency matrices of our stuff.
dataframe = pd.read_csv('CSV/LED_Circuit.csv', index_col=0)
df_big_circuit = pd.read_csv('CSV/Massive_Circuit.csv', index_col=0)
massive_circuit = nx.from_pandas_adjacency(df_big_circuit)
LED_Circuit = nx.from_pandas_adjacency(dataframe)

print(f"Graph with {LED_Circuit.number_of_nodes()} nodes and {LED_Circuit.number_of_edges()} edges")
print("Nodes:", list(LED_Circuit.nodes()))
print("Edges:", list(LED_Circuit.edges()))

# Check Planarity -- we are just double checking that these are correct implementations of our known, planar circuits.
planar_check_LED = nx.is_planar(LED_Circuit)
planar_check_MC = nx.is_planar(massive_circuit)

print("LED circuit planar:", planar_check_LED)
print("Massive Circuit:", planar_check_MC)

# Prepare data for each graph
graphs_data = []

# LED Circuit
led_node_colors = [get_node_color(node) for node in LED_Circuit.nodes()]
led_labels = {node: get_short_label(node) for node in LED_Circuit.nodes()} 
led_pos_start = nx.random_layout(LED_Circuit)
led_pos_end = nx.planar_layout(LED_Circuit) if planar_check_LED else nx.spring_layout(LED_Circuit, k=2, iterations=100)
graphs_data.append({
    'graph': LED_Circuit,
    'colors': led_node_colors,
    'labels': led_labels,
    'pos_start': led_pos_start,
    'pos_end': led_pos_end,
    'title': 'LED Circuit - Untangling to Planar Layout',
    'filename': 'Planar LED Circuit',
    'node_size': 200,
    'font_size': 9,
    'use_legend': True
})

# Massive Circuit
mc_node_colors = [get_node_color(node) for node in massive_circuit.nodes()]
mc_labels = {node: get_short_label(node) for node in massive_circuit.nodes()}
mc_pos_start = nx.random_layout(massive_circuit)
mc_pos_end = nx.planar_layout(massive_circuit) if planar_check_MC else nx.spring_layout(massive_circuit, k=2, iterations=100)
graphs_data.append({
    'graph': massive_circuit,
    'colors': mc_node_colors,
    'labels': mc_labels,
    'pos_start': mc_pos_start,
    'pos_end': mc_pos_end,
    'title': 'Massive Circuit - Untangling to Planar Layout',
    'filename': 'Planar Massive Circuit',
    'node_size': 200,
    'font_size': 7,
    'use_legend': True
})

# Animation Control Variables.
MOVEMENT_FRAMES = 100
PAUSE_FRAMES = 40
num_frames = MOVEMENT_FRAMES + PAUSE_FRAMES # total frames.
animations = [] 

# Create output directory for GIFs
output_dir = os.path.join(os.getcwd(), "Figures")
os.makedirs(output_dir, exist_ok=True)

# Create animations for each graph
for idx, data in enumerate(graphs_data):
    # Create the figure and axes uniquely for each iteration - we reset the frame.
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(QB_DARK_BG)
    ax.set_facecolor(QB_DARK_BG)
    ax.axis('off')
    
    # Build legend dynamically for this specific graph
    legend_elements = build_dynamic_legend(data['graph'])
    
    def animate(frame, graph_data=data, current_ax=ax):
        """
        Animation function called for each frame in our motion.
        """
        # Reset the frame.
        current_ax.clear()
        current_ax.set_facecolor(QB_DARK_BG)
        current_ax.axis('off')

        # Pausing at the end of the animation.
        if frame < MOVEMENT_FRAMES:
            # Movement Phase -- t goes from 0.0 to 1.0
            t = frame / (MOVEMENT_FRAMES - 1)
        else:
            # Pause Phase -- t is held at 1.0 (final position)
            t = 1.0
        
        # Interpolate positions using the improved function
        pos_current = interpolate_positions(graph_data['pos_start'], graph_data['pos_end'], t)
        
        # Draw the graph
        nx.draw_networkx_edges(graph_data['graph'], pos_current, ax=current_ax, 
                              edge_color=QB_DARK_TEXT, width=1.5, alpha=0.7)
        nx.draw_networkx_nodes(graph_data['graph'], pos_current, ax=current_ax,
                              node_color=graph_data['colors'], node_size=graph_data['node_size'])
        nx.draw_networkx_labels(graph_data['graph'], pos_current, graph_data['labels'], ax=current_ax,
                               font_color='#000000', font_size=graph_data['font_size'], 
                               font_weight='bold')
        
        # Add legend
        current_ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
                 facecolor=QB_DARK_BG, edgecolor=QB_DARK_TEXT, labelcolor=QB_DARK_TEXT)
    
    # Create animation
    anim = FuncAnimation(fig, animate, fargs=(data, ax), frames=num_frames, interval=50, repeat=True)
    animations.append(anim)
    
    # Save animation as GIF
    output_path = os.path.join(output_dir, f"{data['filename']} (animation).gif")
    anim.save(output_path, writer='pillow', fps=30)
    print(f"Saved animation: {output_path}")
    
    # Save static image of final frame
    static_image_path = os.path.join(output_dir, f"{data['filename']} (static).png")
    plt.savefig(static_image_path, facecolor=fig.get_facecolor())
    print(f"Saved static image: {static_image_path}")
    
plt.show()