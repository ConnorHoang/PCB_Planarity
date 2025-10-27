"""
A file that tests planarity and visualizes it if it can be scrambled into a planar graph.
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

def setup_matplotlib_backend():
    """
    Automatically detect environment and set appropriate matplotlib backend.
    This ensures compatibility across different environments (WSL2, Docker, local, etc.)
    """
    # Check if we're in a headless environment
    is_headless = (
        os.environ.get('DISPLAY') is None or  # No X11 display
        'SSH_CLIENT' in os.environ or         # SSH session
        'SSH_TTY' in os.environ or            # SSH session
        sys.platform.startswith('linux') and os.path.exists('/.dockerenv')  # Docker
    )
    
    # Check if we're in WSL2
    is_wsl2 = False
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read()
            is_wsl2 = 'microsoft' in version_info.lower() or 'wsl' in version_info.lower()
    except:
        pass
    
    # Backend selection logic
    if is_headless or is_wsl2:
        # Use Agg backend for headless environments
        matplotlib.use('Agg')
        print("Using Agg backend (headless environment detected)")
        return 'headless'
    else:
        # Try GUI backends in order of preference
        gui_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg']
        for backend in gui_backends:
            try:
                matplotlib.use(backend)
                print(f" Using {backend} backend (GUI environment)")
                return 'gui'
            except ImportError:
                continue
        
        # Fallback to Agg if no GUI backend works
        matplotlib.use('Agg')
        print(" Using Agg backend (GUI backends unavailable)")
        return 'headless'

# Setup matplotlib backend
display_mode = setup_matplotlib_backend()

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
        parts = node_name.split('_') #
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

# Load the adjacency matrices of our stuff.
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
circuit1_path = os.path.join(project_root, 'circuits', 'Circuit_1.csv')
circuit2_path = os.path.join(project_root, 'circuits', 'Circuit_2.csv')

dataframe = pd.read_csv(circuit1_path, index_col=0)
df_big_circuit = pd.read_csv(circuit2_path, index_col=0)
LED_Circuit = nx.from_pandas_adjacency(dataframe)
massive_circuit = nx.from_pandas_adjacency(df_big_circuit)

# Check Planarity
planar_check_LED = nx.is_planar(LED_Circuit)
planar_check_MC = nx.is_planar(massive_circuit)

# uncomment if you want to see the boolean value.
# print(planar_check_LED)
# print(planar_check_MC)

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
    'node_size': 200,
    'font_size': 7,
    'use_legend': True
})

# Create legend elements
legend_elements = [
    Patch(facecolor='#E31A1C', label='Power (5V)'),
    Patch(facecolor='#000000', label='Ground (GND)'),
    Patch(facecolor='#1F78B4', label='Switch'),
    Patch(facecolor='#FF7F00', label='LED'),
    Patch(facecolor='#FDBF6F', label='Resistor'),
    Patch(facecolor='#6A3D9A', label='Capacitor'),
    Patch(facecolor='#33A02C', label='Transistor'),
    Patch(facecolor='#CAB2D6', label='Buzzer'),
]

# Animation Control Variables.
MOVEMENT_FRAMES = 100
PAUSE_FRAMES = 40
num_frames = MOVEMENT_FRAMES + PAUSE_FRAMES # total frames.
animations = []
figures = []  # Store figures for static plot saving 

# Create animations for each graph
graph_names = ['LED_Circuit', 'Massive_Circuit']  # Define graph names
for idx, data in enumerate(graphs_data):
    # Create the figure and axes uniquely for each iteration - we reset the frame.
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(QB_DARK_BG)
    ax.set_facecolor(QB_DARK_BG)
    ax.axis('off')
    
    def animate(frame, graph_data=data, current_ax=ax):
        """
        Animation function called for each frame in our motion.
        """
        # Reset the frame.
        current_ax.clear()
        current_ax.set_facecolor(QB_DARK_BG)
        current_ax.axis('on')

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
    # interval is 50 ms.
    anim = FuncAnimation(fig, animate, fargs=(data, ax), frames=num_frames, interval=50, repeat=True)
    animations.append(anim)
    plt.tight_layout()
    
    # Save the animation as a GIF file
    current_graph_name = graph_names[idx]
    output_path = os.path.join(project_root, 'circuit_diagrams', f'{current_graph_name}_animation.gif')
    anim.save(output_path, writer='pillow', fps=20)
    print(f"Animation saved to: {output_path}")
    
    # Store figure for static plot saving
    figures.append(fig)

# Handle output based on environment
if display_mode == 'headless':
    # Save static plots in headless mode
    for i, (fig, graph_name) in enumerate(zip(figures, graph_names)):
        static_output_path = os.path.join(project_root, 'circuit_diagrams', f'{graph_name}_static.png')
        fig.savefig(static_output_path, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to: {static_output_path}")
    print("All plots saved successfully! (Headless mode)")
else:
    # Show plots in GUI mode, but also save them
    for i, (fig, graph_name) in enumerate(zip(figures, graph_names)):
        static_output_path = os.path.join(project_root, 'circuit_diagrams', f'{graph_name}_static.png')
        fig.savefig(static_output_path, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to: {static_output_path}")
    
    print("All plots saved successfully! (GUI mode)")
    print("Displaying plots...")
    plt.show()