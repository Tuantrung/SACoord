import numpy as np


class GreedyCoord:

    def __init__(self, network, ig_node, num_sfcs, num_sfs, flow_weight=1, delay_weight=0):
        self.network = network
        self.ig_node = ig_node
        self.num_sfcs = num_sfcs
        self.num_sfs = num_sfs
        self.flow_weight = flow_weight
        self.delay_weight = delay_weight
        self.output = []
        self.active_node = []
        self.inactive_node = []

    def greedy_schedule(self):
        # scheduling at ingress node
        for node in self.network.nodes.items():
            if node[1]["type"] == "Ingress":
                node[1]["active"] = 1
                self.active_node.append(node)
            else:
                node[1]["active"] = 0
                self.inactive_node.append(node)
            node[1]["weight"] = node[1]["active"] / len(self.ig_node)

        # scheduling in next node
        # for i in range(self.num_sfs - 1):
        for i in range(self.num_sfs):
            self.schedule_next_node()

        pop0_schedule = np.zeros(shape=len(self.network) * self.num_sfcs * self.num_sfs)

        for index, v in enumerate(self.output):
            pop0_schedule[index] += pop0_schedule[index] + v[1]

        return pop0_schedule

    def schedule_next_node(self):
        sum = 0

        for node in self.inactive_node:
            for n in self.active_node:
                if self.network.graph['shortest_paths'][(n[0], node[0])][1] > len(self.network) or node[1]["cap"] == 0:
                    node[1]["weight"] = 0
                    self.inactive_node.remove(node)
                    break
                else:
                    node[1]["weight"] = max(((self.flow_weight + 1) * node[1]['cap'] + (self.delay_weight + 1) /
                                         self.network.graph['shortest_paths'][(n[0], node[0])][1]), node[1]["weight"])
                n[1]["weight"] = 0
            sum += node[1]["weight"]

        # reset active node
        self.active_node.clear()
        self.inactive_node.clear()

        for node in self.network.nodes(data=True):
            if sum == 0:
                print("No solution found")
                break
            self.output.append((node[0], round(node[1]["weight"] / sum, 2)))
            if node[1]["weight"] > 0:
                node[1]["active"] = 1
                self.active_node.append(node)
            else:
                node[1]["active"] = 0
                self.inactive_node.append(node)