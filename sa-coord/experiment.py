from spinterface import SimulatorAction
from siminterface.simulator import Simulator
from rlsp.agents.main import get_base_path

if __name__ == '__main__':
    network_path = '../params/networks/triangle.graphml'
    service_path = '../params/services/abc.yaml'
    sim_config_path = '../params/config/sim_config.yaml'
    test_mode = True
    base_path = get_base_path('', network_path, service_path, sim_config_path)
    config_dir = f'../results/{base_path}'
    simulator = Simulator(network_path, service_path, sim_config_path)
    state = simulator.init(seed=1234)

    placement = {
        'pop0': ['a', 'b', 'c'],
        'pop1': ['b'],
        'pop2': ['a']
    }

    scheduling = {
        'pop0': {
            'sfc_1': {
                'a': {
                    'pop0': 0.4,
                    'pop1': 0.6,
                    'pop2': 0
                },
                'b': {
                    'pop0': 0.6,
                    'pop1': 0.2,
                    'pop2': 0.2
                },
                'c': {
                    'pop0': 0.6,
                    'pop1': 0.2,
                    'pop2': 0.2
                }
            }
        },
        'pop1': {
            'sfc_1': {
                'a': {
                    'pop0': 0.4,
                    'pop1': 0.6,
                    'pop2': 0
                },
                'b': {
                    'pop0': 0.6,
                    'pop1': 0.2,
                    'pop2': 0.2
                },
                'c': {
                    'pop0': 0.6,
                    'pop1': 0.2,
                    'pop2': 0.2
                }
            }
        },
        'pop2': {
            'sfc_1': {
                'a': {
                    'pop0': 0.4,
                    'pop1': 0.6,
                    'pop2': 0
                },
                'b': {
                    'pop0': 0.6,
                    'pop1': 0.2,
                    'pop2': 0.2
                },
                'c': {
                    'pop0': 0.6,
                    'pop1': 0.2,
                    'pop2': 0.2
                }
            }
        }
    }
    # placement = {}
    # scheduling = {}
    dummy_action = SimulatorAction(placement=placement, scheduling=scheduling)
    state = simulator.apply(dummy_action)
    placement = state.placement
    network_stats = state.network_stats
    print(placement)

