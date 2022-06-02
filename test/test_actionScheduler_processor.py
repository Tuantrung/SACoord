from siminterface.simulator import Simulator, SimulatorAction

network_path = '../params/networks/triangle.graphml'
service_path = '../params/services/abc.yaml'
sim_config_path = '../params/config/sim_config.yaml'

placement = {
        'pop0': ['a', 'b', 'c'],
        'pop1': ['a', 'b', 'c'],
        'pop2': ['a', 'b', 'c']
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
# test_action_norm_processor = ActionScheduleProcessor(3, 1, 3)

simulator = Simulator(network_path, service_path, sim_config_path)
simulator.init(344)
action_1 = SimulatorAction(placement=placement, scheduling=scheduling)
simulator_state = simulator.apply(action_1)
print(simulator_state.network['nodes'])

# sfc_list = get_sfc(service_path)
# network, _, _ = read_network(network_path)
# env_limits = EnvironmentLimits(len(network.nodes), sfc_list, len(sfc_list))

#
# action = np.array([0, 0.4, 0.6, 0.6, 0.2, 0.2, 0.6, 0.2, 0.2, 0, 0.4, 0.6, 0.6, 0.2, 0.2, 0.6, 0.2, 0.2,
#                    0, 0.4, 0.6, 0.6, 0.2, 0.2, 0.6, 0.2, 0.2])

# action2 = test_action_norm_processor.process_action(action)

# simulator_wrapper = SASimulatorWrapper(simulator, env_limits)

# state1 = simulator_wrapper.apply(action)

# network_stats_1 = state1.network_stats
# print(network_stats_1)

# random_process = GaussianWhiteNoiseProcess(sigma=0.2, mu=0.0, size=27)
#
# s = action + random_process.sample()
#
# a = test_action_norm_processor.process_action(s)