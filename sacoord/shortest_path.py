from algorithms.shortestPath import get_placement_schedule
from coordsim.reader.reader import get_sfc
from common.common_functionalities import get_ingress_nodes_and_cap
import numpy as np

# network_path = '../res/networks/abilene/abilene-in4-rand-cap0-2.graphml'
# service_path = '../res/service_functions/abc.yaml'
# network, _, _ = read_network(network_path)


def get_solution(network, service_path):
    nodes_list = [node[0] for node in network.nodes(data=True)]
    sfcs = get_sfc(service_path)
    sfc_list = list(sfcs.keys())
    sf_list = sfcs[sfc_list[0]]

    output = []

    ingress_nodes, nodes_cap = get_ingress_nodes_and_cap(network, cap=True)

    placement, schedule = get_placement_schedule(network, nodes_list, sf_list, sfc_list, ingress_nodes, nodes_cap)

    for src in nodes_list:
        for sfc in sfc_list:
            for sf in sf_list:
                for i in range(len(nodes_list)):
                    output.append(schedule[src][sfc][sf][nodes_list[i]])
    return np.array(output)
