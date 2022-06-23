from sacoord.shortest_path import get_solution
from coordsim.reader.reader import read_network, get_sfc


network_path = '../res/networks/abilene/abilene-in4-rand-cap0-2.graphml'
service_path = '../res/service_functions/abc.yaml'
network, _, _ = read_network(network_path)

a = get_solution(network, service_path)

print(a)
