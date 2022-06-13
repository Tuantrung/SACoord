from spinterface import SimulatorAction
from siminterface.simulator import Simulator
from rlsp.agents.main import get_base_path
from coordsim.reader.reader import read_network

if __name__ == '__main__':
    network_path = '../res/networks/5node/5node-in2-rand-cap0-2.graphml'
    # service_path = '../res/service_functions/abc.yaml'
    # sim_config_path = '../res/config/simulator/det-arrival10_det-size001_duration100.yaml'
    # base_path = get_base_path('', network_path, service_path, sim_config_path)
    # config_dir = f'../results/{base_path}'
    # simulator = Simulator(network_path, service_path, sim_config_path)
    # state = simulator.init(seed=1234)
    network, ig_node, eg_node = read_network(network_path)

    # for e in network.edges(data=True):
    #     print(e)
    for n in network.nodes(data=True):
        print(n)

    print(network.graph['shortest_paths'][('pop3', 'pop4')])


    # placement = {
    #     'pop0': ['a'],
    #     'pop1': ['b'],
    #     'pop2': ['c'],
    #     'pop4': ['a']
    # }
    #
    # scheduling = {
    #     'pop0': {
    #         'sfc_1': {
    #             'a': {
    #                 'pop0': 0.2,
    #                 'pop1': 0.2,
    #                 'pop2': 0,
    #                 'pop3': 0.4,
    #                 'pop4': 0.4
    #             },
    #             'b': {
    #                 'pop0': 0.6,
    #                 'pop1': 0.2,
    #                 'pop2': 0.2
    #             },
    #             'c': {
    #                 'pop0': 0.6,
    #                 'pop1': 0.2,
    #                 'pop2': 0.2
    #             }
    #         }
    #     },
    #     'pop1': {
    #         'sfc_1': {
    #             'a': {
    #                 'pop0': 0.4,
    #                 'pop1': 0.6,
    #                 'pop2': 0
    #             },
    #             'b': {
    #                 'pop0': 0.6,
    #                 'pop1': 0.2,
    #                 'pop2': 0.2
    #             },
    #             'c': {
    #                 'pop0': 0.6,
    #                 'pop1': 0.2,
    #                 'pop2': 0.2
    #             }
    #         }
    #     },
    #     'pop2': {
    #         'sfc_1': {
    #             'a': {
    #                 'pop0': 0.4,
    #                 'pop1': 0.6,
    #                 'pop2': 0
    #             },
    #             'b': {
    #                 'pop0': 0.6,
    #                 'pop1': 0.2,
    #                 'pop2': 0.2
    #             },
    #             'c': {
    #                 'pop0': 0.6,
    #                 'pop1': 0.2,
    #                 'pop2': 0.2
    #             }
    #         }
    #     }
    # }
    # placement = {}
    # scheduling = {}
    # dummy_action = SimulatorAction(placement=placement, scheduling=scheduling)
    # state = simulator.apply(dummy_action)
    # # placement = state.placement
    # traffic = state.traffic
    # # # network_stats = state.network_stats
    # # print(placement)
    # # print(traffic['pop1']['sfc1'][''])
    # print(traffic)