"""
This file checks the planarity of a graph. When the graph is non-planar,
the minimum number of nodes nedded to make the graph planr will  be found.

The code will also visualize the graph and the process of making it planar.
"""

import networkx as nx
from networkx.algorithms import minors
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import sys

# Get the directory of this script and construct path relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "circuits", "non_planar - Sheet1.csv");


get_matrix = np.loadtxt(csv_path, delimiter=',')  # Skip header row

graph = nx.from_numpy_array(get_matrix);

print(graph.edges());


if nx.is_planar(graph):
    print("The graph is planar");
else:
    print("The graph is not planar");

    subgraph = embedding.copy(graph);
    print(list(subgraph.edges()));
    
