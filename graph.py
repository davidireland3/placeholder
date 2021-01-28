"""
This file contains the graph class used throughout the files.
"""

import numpy as np
import random
import torch


class Graph:
    """
    A class to store graph data. Input is expected to be a numpy array of dimension (n_edges,2). Each row should
    correspond to an edge. If the first row is [0,1] then this means that node 0 points to node 1. If your array only
    has each edge once, i.e. if your graph is supposed to be undirectional but you only have [0,1] in your array but not
    [1,0], then one_way should be set to True. If your graph is directional then leave one_way as False. original_graph
    is an argument that should be passed if subgraph is True - it is the original graph that we are making the subgraph
    from and enables us to preserve edge features, otherwise they are randomly initialised.
    """

    def __init__(self, graph_np, one_way=False, subgraph=False, original_graph=None):
        self.transform = {}  # dictionary for performing the transformation, will remain empty if it is not needed.
        # keys are the original node id and values are the new node id.
        self.inverse_transformation = {}  # this is the inverse of the above. Mapping is a bijection.
        self.graph_np = self.get_node_orderings(graph_np)
        if one_way:
            first_col = graph_np[:, 0].reshape((graph_np.shape[0], 1))
            second_col = graph_np[:, 1].reshape((graph_np.shape[0], 1))
            hold = np.concatenate((second_col, first_col), axis=1)
            self.graph_np = np.concatenate((self.graph_np, hold), axis=0)

        self.nodes = set(self.graph_np[:, 0])
        self.num_nodes = len(self.nodes)
        self.neighbours = self.get_neighbours()
        if subgraph is False:
            self.edge_weights = self.get_edge_weights()
        else:
            self.edge_weights = self.get_original_edge_weights(graph_np, original_graph)

    def get_neighbours(self):
        neighbours = {}
        for node in self.nodes:
            neighbours[node] = []
        for row in range(self.graph_np.shape[0]):
            node = self.graph_np[row, 0]
            neighbour = self.graph_np[row, 1]
            neighbours[node].append(neighbour)
        return neighbours

    def get_edge_weights(self):
        '''
        This method assigns edge weights corresponding to the acceptance probabilities from the GCOMB paper using one of
        the models they use, a random probability drawn from [0.1,0.01,0.001].
        '''
        edge_weights = {}
        for node in self.neighbours.keys():
            for neighbour in self.neighbours[node]:
                try:
                    edge_weights[(node, neighbour)] = edge_weights[(neighbour, node)]
                except KeyError:
                    edge_weights[(node, neighbour)] = float(np.random.choice([0.001, 0.01, 0.1], size=1))
        return edge_weights

    def get_subset(self, subset_nodes):  # accepts a percentage
        subset_edges = []
        subset_nodes = set(subset_nodes)
        for _ in range(self.graph_np.shape[0]):
            node = self.graph_np[_, 0]
            neighbour = self.graph_np[_, 1]
            if node in subset_nodes and neighbour in subset_nodes:
                subset_edges.append([node, neighbour])
        return Graph(np.array(subset_edges))

    def get_split(self, splits=[0.7, 0.15, 0.15]):  # splits should be a list for propn you want as test, train, val
        '''
        This method splits the graph into test graph, train graph, val graph with pre set %ages from the GCOMB paper.
        '''
        test_size = int(np.floor(self.num_nodes * splits[0]))
        train_size = int(np.floor(self.num_nodes * splits[1]))
        val_size = self.num_nodes - test_size - train_size

        node_order = self.nodes
        random.shuffle(node_order)
        n = len(node_order)

        test_nodes = node_order[0:test_size]
        train_nodes = node_order[test_size:(n - val_size + 1)]
        val_nodes = node_order[(n - val_size):]

        test_graph = self.get_subset(test_nodes)
        train_graph = self.get_subset(train_nodes)
        val_graph = self.get_subset(val_nodes)

        return test_graph, train_graph, val_graph

    def get_node_orderings(self, graph_array):
        """
        this module is for when taking a subset of a real graph to make sure all the nodes are labelled 0:n to comply
        with DGL, otherwise DGL fills in random nodes. E.g. if we had a node set of {1,2,5} then DGL would make a graph
        where the node set would be {1,2,3,4,5} so this performs a re-labelling to {0,1,2} with 5 mapped to 0.
        """
        nodes = set(graph_array.flatten())  # gets unique nodes
        n = len(nodes)  # length of nodes
        not_in_node_set = []  # indexes not in our node set

        for i in range(n):
            if i not in nodes:
                not_in_node_set.append(i)

        for node in nodes:
            if node >= n:
                new_index = not_in_node_set.pop()
                self.transform[node] = new_index
                self.inverse_transformation[new_index] = node

        for row in range(graph_array.shape[0]):
            x = graph_array[row, 0]
            y = graph_array[row, 1]

            if x >= n:
                graph_array[row, 0] = self.transform[x]

            if y >= n:
                graph_array[row, 1] = self.transform[y]

        return graph_array

    def get_features(self):
        """
        This module returns a dictionary that has keys of the node label and values are the features (which currently
        is the outgoing edge weights). Use values to make a tensor for Pytorch.
        """
        feats = {}
        for node in sorted(self.nodes):
            feats[node] = 0
        for edge in self.edge_weights.keys():
            feats[edge[0]] += self.edge_weights[edge]
        return feats

    def return_edges(self):
        """
        this method simply returns a list of the root nodes and a list of where they point to. Complies with how
        DGL accepts inputs to make the graph
        """
        u = self.graph_np[:, 0]
        v = self.graph_np[:, 1]
        return u, v

    def k_hop_neighbours(self, node, K, graph):
        """
        This method returns a graph class object which is the sub-graph of the k-hop neighbours from the root node.
        Currently this only accepts one root node, but it should be easy to extend to multiple. Edges are preserved from
        the original graph if and only if the source and destination nodes are both in the k-hop neighbour set.
        """
        edges = []
        edge_set = set()  # for O(1) lookup.
        nodes = {node}
        k_neighbours = {1: [self.neighbours[node]]}
        for k in range(2, K + 1):
            k_neighbours[k] = []

        # this nested for loops get the nodes which are the k-hop neighbours

        for k in range(1, K + 1):

            for neighbours_list in k_neighbours[k]:

                for neighbour in neighbours_list:

                    if neighbour not in nodes:

                        nodes.add(neighbour)

                        if k is not K:
                            k_neighbours[k + 1].append(self.neighbours[neighbour])

        # now we need to append the edges from the original graph for all nodes in the nodes set

        for node in nodes:

            for neighbour in self.neighbours[node]:

                if neighbour in nodes:

                    if (node, neighbour) not in edge_set and (neighbour, node) not in edge_set:
                        edges.append([node, neighbour])
                        edges.append([neighbour, node])
                        edge_set.add((node, neighbour))
                        edge_set.add((neighbour, node))

        return Graph(np.array(edges), False, True, graph)

    def get_original_edge_weights(self, edges, original_graph):
        """
        This method will maintain the edge-weights from the original graph when we obtain the subgraph from k-hop
        neighbours.
        """
        e_weights = {}
        for edge in edges:
            e1, e2 = edge
            if e1 in self.inverse_transformation.keys():
                e1 = self.inverse_transformation[e1]
            if e2 in self.inverse_transformation.keys():
                e2 = self.inverse_transformation[e2]
            e_weights[tuple(edge)] = original_graph.edge_weights[(e1, e2)]
        return e_weights


def make_one_way(edges):
    """
    This module makes a graph 'one way'. If the dataset downloaded is bi-directional (e.g. if [1,100] is a row in the
    array then [100,1] will also be in there, this deletes the duplicate row. This is to make it more consistent when
    passing the data into the Graph class.
    """
    new_edges = []
    new_edges_set = set()
    for row in range(edges.shape[0]):
        root = edges[row][0]
        neighbour = edges[row][1]
        if (neighbour, root) not in new_edges_set:
            new_edges.append([root, neighbour])
            new_edges_set.add((root, neighbour))
    return np.array(new_edges)


def expected_spread(seed_nodes, graph, mc_its=10000):  # node_neighbours is a dic keys = nodes, vals = neighb
    node_neighbours = graph.neighbours
    edge_weights = graph.edge_weights

    if type(seed_nodes) is set:
        seed_nodes = list(seed_nodes)

    if not seed_nodes:
        return 0

    spreads = []
    for _ in range(mc_its):  # start of MC loop
        activated_nodes = set(seed_nodes)  # initial activated nodes, i.e. the seed nodes
        nodes_to_consider = []  # set up empty list of nodes to consider
        for seed_node in seed_nodes:  # loops over seed node neighbours to determine which nodes we consider for activ.
            neighbours = node_neighbours[seed_node]
            for neighbour in neighbours:
                nodes_to_consider.append((seed_node, neighbour))

        while nodes_to_consider:  # whilst we have nodes to consider

            root_node, current_node = nodes_to_consider.pop()  # pop one off the list

            p = edge_weights[(root_node, current_node)]  # get the edge weight corresponding to the current node
            if np.random.uniform() < p and current_node not in activated_nodes:  # if activated and not already in then
                # add to activated nodes, and append the activated nodes neighbours to also consider

                activated_nodes.add(current_node)

                for neighbour in node_neighbours[current_node]:
                    nodes_to_consider.append((current_node, neighbour))

        spread = len(activated_nodes) / graph.num_nodes * 100  # calculates spread
        spreads.append(spread)

    return np.mean(spreads)


# def k_hop_neighbours(self, node, K):
#
#     edges = []
#     edge_set = set()  # this is for O(1) lookup
#     k_neighbours = {1: [(node, self.neighbours[node])]}
#     for i in range(2, K + 1):
#         k_neighbours[i] = []
#
#     for k in range(1, K + 1):
#
#         for i in range(len(k_neighbours[k])):
#
#             root = k_neighbours[k][i][0]
#
#             for neighbour in k_neighbours[k][i][1]:
#
#                 if (root, neighbour) not in edge_set and (neighbour, root) not in edge_set:
#
#                     # add the edges to the relevant objects
#                     edges.append([root, neighbour])
#                     edges.append([neighbour, root])
#                     edge_set.add((root, neighbour))
#                     edge_set.add((neighbour, root))
#
#                     # now add the neighbour and its list of neighbours to the dictionary if k is not K
#                     if k is not K:
#                         k_neighbours[k + 1].append((neighbour, self.neighbours[neighbour]))
#
#     return Graph(np.array(edges))

def get_reward(node1, node2, graph: Graph):
    node1_neighbours = set(graph.neighbours[node1])
    node2_neighbours = set(graph.neighbours[node2])

    reward = 0

    if len(node1_neighbours) < len(node2_neighbours):

        for neighbour in node1_neighbours:

            if neighbour in node2_neighbours:
                reward += (graph.edge_weights[(node1, neighbour)] + graph.edge_weights[(node2, neighbour)]) / 2

    else:

        for neighbour in node2_neighbours:

            if neighbour in node1_neighbours:
                reward += (graph.edge_weights[(node1, neighbour)] + graph.edge_weights[(node2, neighbour)]) / 2

    if node1 in node2_neighbours:

        reward += (graph.edge_weights[(node1, node2)] + graph.edge_weights[(node2, node1)]) / 2

    return reward


def get_edge_features(dgl_graph, graph):

    u, v = dgl_graph.edges()
    u = u.numpy().reshape((u.shape[0], 1))
    v = v.numpy().reshape((v.shape[0], 1))
    edge_list = np.concatenate((u, v), axis=1)
    features = np.zeros((edge_list.shape[0], 1))
    for edge, count in zip(edge_list, range(edge_list.shape[0])):

        if count == 0:
            features[count] = graph.edge_weights[tuple(edge)]

        else:
            features[count] = graph.edge_weights[tuple(edge)]

    return torch.from_numpy(features).type(torch.FloatTensor)


def get_node_features(graph):
    features = torch.FloatTensor(list(graph.get_features().values())).reshape((graph.num_nodes, 1))
    return features
