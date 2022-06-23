from coordsim.reader.reader import read_network
import numpy as np

flow_weight = 1
delay_weight = 0
network_path = '../res/networks/5node/5node-in2-rand-cap0-2.graphml'
network, ig_node, _ = read_network(network_path)

output = []
active_node = []
inactive_node = []

for node in network.nodes.items():
    if node[1]["type"] == "Ingress":
        node[1]["active"] = 1
    else:
        node[1]["active"] = 0
    node[1]["weight"] = node[1]['active']/len(ig_node)

    if node[1]["active"] == 1:
        active_node.append(node)
    else:
        inactive_node.append(node)

    output.append((node[0], node[1]['weight']))

sum = 0

for node in inactive_node:
    for n in active_node:
        if network.graph['shortest_paths'][(n[0], node[0])][1] > 3:
            node[1]['weight'] = 0
            inactive_node.remove(node)
        else:
            node[1]['weight'] = max(((flow_weight + 1) * node[1]['cap'] + (delay_weight + 1) / network.graph['shortest_paths'][(n[0], node[0])][1]), node[1]['weight'])
        n[1]['weight'] = 0
    sum += node[1]['weight']

# reset active node
active_node.clear()
inactive_node.clear()

for node in network.nodes(data=True):
    output.append((node[0], round(node[1]['weight'] / sum, 2)))
    if node[1]['weight'] > 0:
        node[1]['active'] = 1
        active_node.append(node)
    else:
        node[1]['active'] = 0
        inactive_node.append(node)

# reset sum
sum = 0

for node in inactive_node:
    for n in active_node:
        if network.graph['shortest_paths'][(n[0], node[0])][1] > 3:
            node[1]['weight'] = 0
            inactive_node.remove(node)
        else:
            node[1]['weight'] = max(((flow_weight + 1) * node[1]['cap'] + (delay_weight + 1) / network.graph['shortest_paths'][(n[0], node[0])][1]), node[1]['weight'])
        n[1]['weight'] = 0
    sum += node[1]['weight']

# reset active node
active_node.clear()
inactive_node.clear()

for node in network.nodes(data=True):
    output.append((node[0], round(node[1]['weight'] / sum, 2)))
    if node[1]['weight'] > 0:
        node[1]['active'] = 1
        active_node.append(node)
    else:
        node[1]['active'] = 0
        inactive_node.append(node)

pop0_schedule = np.zeros(shape=15)

for index, v in enumerate(output):
    pop0_schedule[index] += pop0_schedule[index] + v[1]











