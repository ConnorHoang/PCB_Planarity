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
import random
from itertools import combinations
import random

# Get the directory of this script and construct path relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
#csv_path = os.path.join(project_root, "circuits", "non_planar - Sheet1.csv");
csv_path = os.path.join(project_root, "circuits", "layered_circuit.csv");


#below code gets eliminated by line 35
adjacency_matrix = pd.read_csv(csv_path, index_col=0)
adjacency_matrix.index = adjacency_matrix.columns 

graph = nx.from_pandas_adjacency(adjacency_matrix)

print(graph.nodes())
print(graph.edges())
###########################################################

graph = nx.complete_bipartite_graph(3, 3)

# ---- helpers for estimating crossings ----
def _ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(p1, p2, p3, p4):
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))

def count_crossings(G, pos):
    edges = [(u, v) for u, v in G.edges() if u != v]
    crossings = 0
    for (u, v), (x, y) in combinations(edges, 2):
        if set((u, v)) & set((x, y)):
            continue
        p1, p2, p3, p4 = pos[u], pos[v], pos[x], pos[y]
        if segments_intersect(p1, p2, p3, p4):
            crossings += 1
    return crossings

def guess_crossing(G, reps=30):
    best = None
    best_pos = None
    for _ in range(reps):
        seed = random.randint(0, 1_000_000)
        pos = nx.spring_layout(G, seed=seed)
        crossings = count_crossings(G, pos)
        if best is None or crossings < best:
            best = crossings
            best_pos = pos
    return best, best_pos


################################

is_planar , embedding = nx.check_planarity(graph);

if  is_planar:
    print("The graph is planar");
else:
    print("The graph is not planar");
    best, lowest_graph = guess_crossing(graph)
    print(best)
    print(lowest_graph)
    # lowest_graph is a position dict; draw the graph using these positions
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(graph, pos=lowest_graph, with_labels=True, node_color='lightblue', edgecolors='black')
    out_dir = os.path.join(project_root, 'circuit_diagrams')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'nonplanar_best_layout.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved best layout to: {out_path}")
    plt.close()
    

