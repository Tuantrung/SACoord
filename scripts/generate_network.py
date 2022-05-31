import networkx as nx
import random


def gen_new_network(network_file, min_cap, max_cap):
    G = nx.Graph()

    G.add_node(0, id=0, Latitude=40.71427, Longitude=-74.00597, NodeCap=random.randint(min_cap, max_cap), NodeType='Normal')
    G.add_node(1, id=1, Latitude=41.85003, Longitude=-87.65005, NodeCap=random.randint(min_cap, max_cap), NodeType='Normal')
    G.add_node(2, id=2, Latitude=38.89511, Longitude=-77.03637, NodeCap=random.randint(min_cap, max_cap), NodeType='Normal')
    G.add_node(3, id=3, Latitude=33.749, Longitude=-84.38798, NodeCap=random.randint(min_cap, max_cap), NodeType='Normal')
    G.add_node(4, id=4, Latitude=39.76838, Longitude=-86.15804, NodeCap=random.randint(min_cap, max_cap), NodeType='Normal')

    G.add_edge(0, 1, LinkFwdCap=random.randint(500, 1000))
    G.add_edge(0, 2, LinkFwdCap=random.randint(500, 1000))
    G.add_edge(1, 4, LinkFwdCap=random.randint(500, 1000))
    G.add_edge(2, 3, LinkFwdCap=random.randint(500, 1000))
    G.add_edge(3, 4, LinkFwdCap=random.randint(500, 1000))
    G.add_edge(1, 3, LinkFwdCap=random.randint(500, 1000))
    G.add_edge(4, 2, LinkFwdCap=random.randint(500, 1000))

    nx.write_graphml(G, path=network_file)


def set_ingress(network_file, new_network_file, ingress_id):
    """Create copy of network with node ingress_id as ingress; keep existing ingress nodes"""
    network = nx.read_graphml(network_file)
    print(f"Current nodes of {network_file}:")
    for v in network.nodes(data=True):
        print(v)

    # set new node attributes, print again, and save to new file
    network.nodes[ingress_id]['NodeType'] = 'Ingress'
    print(f"\nNew nodes; saved to {new_network_file}:")
    for v in network.nodes(data=True):
        print(v)
    nx.write_graphml(network, new_network_file)

    return new_network_file


def main():
    network_file = '../res/networks/5node/5node-in0-rand-cap0-2.graphml'
    new_network_file = '../res/networks/5node/5node-in1-rand-cap0-2.graphml'
    new_network_file_1 = '../res/networks/5node/5node-in2-rand-cap0-2.graphml'
    # gen_new_network(network_file, 0, 2)
    set_ingress(network_file, new_network_file, '1')
    set_ingress(new_network_file, new_network_file_1, '4')


if __name__ == '__main__':
    main()
