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


is_planar , embedding = nx.check_planarity(graph);

if  is_planar:
    print("The graph is planar");
else:
    print("The graph is not planar");


def guess_crossing(graph):
    reps = 30;
    layout = nx.spring_layout(graph);
    lowest = None;
    lowest_graph = None;

    for i in range(reps):
        seed = random.randint(0, 1000000);
        rep_pos = layout(graph,seed)

        crossings = count_crossings(graph, rep_pos);
        if best is None or crossings < best:
            best = crossings;
            lowest_graph = rep_pos;
    return best,lowest_graph;






#chat

def _ccw(A, B, C):
    # return True if points A,B,C are counter-clockwise
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(p1, p2, p3, p4):
    # segment p1-p2 intersects p3-p4 (proper intersection)
    # excludes touching at endpoints (which for graph crossings we usually ignore if they share nodes)
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def count_crossings(G, pos):
    """
    G : networkx.Graph (undirected, simple)
    pos : dict node -> (x,y) coordinates
    returns: integer number of pairwise edge crossings in that drawing
    """
    # prepare list of edges as pairs of endpoints coordinates, ignoring multi-edges/selfloops
    edges = [(u, v) for u, v in G.edges() if u != v]
    crossings = 0
    # check every unordered pair of edges
    for (u, v), (x, y) in combinations(edges, 2):
        # skip if edges share an endpoint (not a crossing)
        if set((u, v)) & set((x, y)):
            continue
        p1, p2, p3, p4 = pos[u], pos[v], pos[x], pos[y]
        if segments_intersect(p1, p2, p3, p4):
            crossings += 1
    return crossings


    

