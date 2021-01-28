import multiprocessing as mp
from graph import expected_spread
import pickle
import itertools
import os

with open("test_graph", mode="rb") as f:
    graph = pickle.load(f)

SEED_NODES = []
for pair in itertools.combinations(graph.nodes, 3):
    SEED_NODES.append(pair)


def get_expected_spread(seed_nodes):
    print(f"Worker {os.getpid()} calculating spread of {seed_nodes}")
    spread = expected_spread(seed_nodes, graph)
    return [tuple(seed_nodes), spread]


if __name__ == "__main__":

    with mp.Pool(processes=20) as pool:
        output = pool.map(get_expected_spread, SEED_NODES)

    spreads = {}
    for item in output:
        spreads[item[0]] = item[1]

    with open("spreads", mode="wb") as f:
        pickle.dump(spreads, f)
