from collections import defaultdict
from coordsim.reader.reader import read_network

network_path = '../res/networks/abilene/abilene-in1-rand-cap0-2.graphml'


def get_closest_neighbours(network, nodes_list):
    """
    Finding the closest neighbours to each node in the network. For each node of the network we maintain a list of
    neighbours sorted in increasing order of distance to it.
    params:
        network: A networkX graph
        nodes_list: a list of nodes in the Network
    Returns:
         closest_neighbour: A dict containing lists of closest neighbour to each node in the network sorted in
                            increasing order to distance.
    """

    all_pair_shortest_paths = network.graph['shortest_paths']
    closest_neighbours = defaultdict(list)
    for source in nodes_list:
        neighbours = defaultdict(int)
        for dest in nodes_list:
            if source != dest:
                delay = all_pair_shortest_paths[(source, dest)][1]
                neighbours[dest] = delay
        sorted_neighbours = [k for k, v in sorted(neighbours.items(), key=lambda item: item[1])]
        closest_neighbours[source] = sorted_neighbours
    return closest_neighbours


network, _, _ = read_network(network_path)
node_list = [node[0] for node in network.nodes(data=True)]

closest_neighbour = get_closest_neighbours(network, node_list)
print(closest_neighbour['pop0'][1])
