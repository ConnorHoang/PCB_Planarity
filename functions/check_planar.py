import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load the adjacency matrix - LED circuit (1)
dataframe = pd.read_csv('/Users/sbloomuel/GitHub/PCB_Planarity/circuits/Circuit_1.csv', index_col=0)
df_big_circuit = pd.read_csv("/Users/sbloomuel/GitHub/PCB_Planarity/circuits/Circuit_2.csv", index_col=0)
massive_circuit = nx.from_pandas_adjacency(df_big_circuit)
LED_Circuit = nx.from_pandas_adjacency(dataframe)

print(f"Graph with {LED_Circuit.number_of_nodes()} nodes and {LED_Circuit.number_of_edges()} edges")
print("Nodes:", list(LED_Circuit.nodes()))
print("Edges:", list(LED_Circuit.edges()))

# Create the K(3,3) subgraph
non_planar_Kthree = nx.complete_bipartite_graph(3, 3)

# Check Planarity
planar_check_LED = nx.is_planar(LED_Circuit)
planar_check_Kthree = nx.is_planar(non_planar_Kthree)
planar_check_MC = nx.is_planar(massive_circuit)

print("LED circuit planar:", planar_check_LED)
print("K3,3 planar:", planar_check_Kthree)
print("Massive Circuit:", planar_check_MC)

# Plot LED Circuit
plt.figure("LED Circuit (Planar)")
nx.draw(LED_Circuit, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("LED Circuit (Planar)")
plt.show(block=False)

# Plot the K3,3
plt.figure("K(3,3) Nonplanar Graph")
nx.draw(non_planar_Kthree, with_labels=True, node_color='lightgreen', edge_color='gray')
plt.title("K(3,3) Graph (Non-Planar)")
plt.show()

# Plot the MC
plt.figure("Massive Circuit Graph")
nx.draw(massive_circuit, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Massive Circuit Graph")
plt.show()