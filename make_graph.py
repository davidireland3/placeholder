import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph import Graph
import pickle

G = nx.random_geometric_graph(60, 0.2, seed=2)
# nx.draw(G, with_labels=True)
# plt.show()

edges_one_way = list(G.edges())
graph = Graph(np.array(edges_one_way), one_way=True)
with open("test_graph", mode="wb") as f:
    pickle.dump(graph, f)
