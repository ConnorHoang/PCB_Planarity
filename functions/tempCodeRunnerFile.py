"""
A file that tests planarity and visualizes it if it can be scrambled into a planar graph.
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import numpy as np

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
dataframe = pd.read_csv('/Users/sbloomuel/GitHub/PCB_Planarity/circuits/Circuit_1.csv', index_col=0)
df_big_circuit = pd.read_csv("/Users/sbloomuel/GitHub/PCB_Planarity/circuits/Circuit_2.csv", index_col=0)
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

# FIX: Restore the random seed for reproducible layouts and ensuring different starting positions for both graphs
np.random.seed(42)

# LED Circuit DATA SETUP
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

# Massive Circuit DATA SETUP
mc_node_colors = [get_node_color(node) for node in massive_circuit.nodes()]
mc_labels = {node: get_short_label(node) for node in massive_circuit.nodes()}
mc_pos_start = nx.random_layout(massive_circuit) # This will produce a different random layout than the LED circuit
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

# FIX: List to hold animation objects and keep them from being garbage collected
animations = [] 

# Create animations for each graph
for idx, data in enumerate(graphs_data):
    # CRITICAL: Create the figure and axes uniquely for each iteration
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(QB_DARK_BG)
    ax.set_facecolor(QB_DARK_BG)
    ax.axis('off')
    
    # FIX: Use a lambda to force the immediate binding of 'data' and 'ax' for THIS iteration.
    # This resolves the late-binding closure issue where all animations were pointing to the final loop values.
    # The inner function is defined to take 'frame' and explicitly uses the immediately bound 'd' and 'a'.
    
    # Define the animation function structure outside the loop (for clarity) or rely on a lambda for binding.
    # We will use the lambda approach combined with an explicit function to ensure robustness.
    
    def create_animate_func(data_instance, ax_instance):
        """
        Creates a custom animate function instance for the current graph and axis.
        This pattern correctly captures the instance of 'data' and 'ax' from the loop.
        """
        def animate(frame):
            """
            Animation function called for each frame in our motion.
            """
            # Reset the frame.
            ax_instance.clear()
            ax_instance.set_facecolor(QB_DARK_BG)
            ax_instance.axis('on')

            # Pausing at the end of the animation.
            if frame < MOVEMENT_FRAMES:
                # Movement Phase -- t goes from 0.0 to 1.0
                t = frame / (MOVEMENT_FRAMES - 1)
            else:
                # Pause Phase -- t is held at 1.0 (final position)
                t = 1.0
            
            # Interpolate positions using the improved function
            pos_current = interpolate_positions(data_instance['pos_start'], data_instance['pos_end'], t)
            
            # Draw the graph
            nx.draw_networkx_edges(data_instance['graph'], pos_current, ax=ax_instance, 
                                  edge_color=QB_DARK_TEXT, width=1.5, alpha=0.7)
            nx.draw_networkx_nodes(data_instance['graph'], pos_current, ax=ax_instance,
                                  node_color=data_instance['colors'], node_size=data_instance['node_size'])
            nx.draw_networkx_labels(data_instance['graph'], pos_current, data_instance['labels'], ax=ax_instance,
                                   font_color=QB_DARK_TEXT, font_size=data_instance['font_size'], 
                                   font_weight='bold')
            
            # Add title with progress
            if t < 1.0:
                progress = int(t * 100)
                title_text = f"{data_instance['title']} (Moving: {progress}%)"
            else:
                title_text = f"{data_instance['title']} (Planar Layout Reached - Paused)"
                
            ax_instance.set_title(title_text, color=QB_DARK_TEXT, fontsize=14, pad=20)
            
            # Add legend
            ax_instance.legend(handles=legend_elements, loc='upper left', framealpha=0.9,
                     facecolor=QB_DARK_BG, edgecolor=QB_DARK_TEXT, labelcolor=QB_DARK_TEXT)
        
        return animate

    # Get the correctly bound animate function for this figure
    animate_func = create_animate_func(data, ax)

    # Create animation
    # The interval remains 50ms, the total frame count is now 140 (100 movement + 40 pause).
    # NOTE: The FuncAnimation is passed the correctly bound function. No need for fargs.
    anim = FuncAnimation(fig, animate_func, frames=num_frames, interval=50, repeat=True)
    
    # FIX: Store the animation object. This is the crucial step to prevent the first plot 
    # from being deleted/suppressed before the second one is shown.
    animations.append(anim)
    
    plt.tight_layout()

# FIX: Call plt.show() once outside the loop to display all figures simultaneously.
plt.show()