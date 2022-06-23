from sacoord.greedy import GreedyCoord
from coordsim.reader.reader import read_network

flow_weight = 1
delay_weight = 0
network_path = '../res/networks/abilene/abilene-in4-rand-cap0-2.graphml'
network1, ig_node, _ = read_network(network_path)
num_sfcs = 1
num_sfs = 3

a = GreedyCoord(network1, ig_node, num_sfcs, num_sfs, flow_weight, delay_weight)
print(a.greedy_schedule())
