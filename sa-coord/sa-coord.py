import logging
from simanneal import Annealer
from rl.random import GaussianWhiteNoiseProcess
from siminterface.simulator import Simulator
from coordsim.reader.reader import read_network, get_sfc, get_sf, network_diameter
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from sa_interface import SASimulatorWrapper

from rlsp.envs.environment_limits import EnvironmentLimits

import numpy as np
from spinterface import SimulatorState

logger = logging.getLogger(__name__)

sigma = 0.2
mu = 0.0
network_path = '../params/networks/triangle.graphml'
service_path = '../params/services/abc.yaml'
sim_config_path = '../params/config/sim_config.yaml'


class CoordProblem(Annealer):
    """
    Implement annealer with Coord Problem.
    """
    simulator_state : SimulatorState

    # pass extra data into the constructor
    def __init__(self, simulator, state, initial_action, network_file, service_file, flow_weight, delay_weight):
        self.simulator = simulator
        self.network_file = network_file
        self.action_processor = ActionScheduleProcessor(num_nodes, num_sfcs, num_sfs)

        self.action = initial_action
        self.curr_simulator_state = state

        self.network, _, _ = read_network(self.network_file)
        self.network_diameter = network_diameter(self.network)
        self.sfc_list = get_sfc(service_file)
        self.sf_list = get_sf(service_file)
        self.env_limits = EnvironmentLimits(len(self.network.nodes), self.sfc_list, 1)

        self.flow_weight = flow_weight
        self.delay_weight = delay_weight

        self.simulator_wrapper = SASimulatorWrapper(self.simulator, self.env_limits)

        self.min_delay, self.max_delay = self.min_max_delay()
        super(CoordProblem, self).__init__(state)

    def move(self):
        """Add Gaussian White Noise to action"""
        initial_energy = self.energy()
        random_process = GaussianWhiteNoiseProcess(sigma=sigma, mu=mu, size=27)
        action = action_processor.process_action(self.action + random_process.sample())

        self.simulator_state = self.simulator_wrapper.apply(action)

        return self.energy() - initial_energy

    def energy(self):
        """Calculates the values of objective function"""
        e = self.calculate_reward(self.curr_simulator_state)
        return e

    def min_max_delay(self):
        """Return the min and max e2e-delay for the current network topology and SFC. Independent of capacities."""
        vnf_delays = sum([sf['processing_delay_mean'] for sf in self.sf_list.values()])
        # min delay = sum of VNF delays (corresponds to all VNFs at ingress)
        min_delay = vnf_delays
        # max delay = VNF delays + num_vnfs * network diameter (corresponds to max distance between all VNFs)
        max_delay = vnf_delays + len(self.sf_list) * self.network_diameter
        logger.info(f"min_delay: {min_delay}, max_delay: {max_delay}, diameter: {self.network_diameter}")
        return min_delay, max_delay

    def get_flow_reward(self, simulator_state):
        """Calculate and return both success ratio and flow reward"""
        # calculate ratio of successful flows in the last run
        cur_succ_flow = simulator_state.network_stats['run_successful_flows']
        cur_drop_flow = simulator_state.network_stats['run_dropped_flows']
        succ_ratio = 0
        flow_reward = 0
        if cur_succ_flow + cur_drop_flow > 0:
            succ_ratio = cur_succ_flow / (cur_succ_flow + cur_drop_flow)
            # use this for flow reward instead of succ ratio to use full [-1, 1] range rather than just [0,1]
            flow_reward = (cur_succ_flow - cur_drop_flow) / (cur_succ_flow + cur_drop_flow)
        return succ_ratio, flow_reward

    def get_delay_reward(self, simulator_state, succ_ratio):
        """Return avg e2e delay and delay reward"""
        # get avg. e2e delay in last run and calculate delay reward
        delay = simulator_state.network_stats['run_avg_end2end_delay']
        # ensure the delay is at least min_delay/VNF delay. may be lower if no flow was successful
        delay = max(delay, self.min_delay)
        # require some flows to be successful for delay to have any meaning; init to -1
        if succ_ratio == 0:
            delay_reward = -1
        else:
            # subtract from min delay = vnf delay;
            # to disregard VNF delay, which cannot be affected and may already be larger than the diameter
            delay_reward = ((self.min_delay - delay) / self.network_diameter) + 1
            delay_reward = np.clip(delay_reward, -1, 1)
        return delay, delay_reward

    def calculate_reward(self, simulator_state: SimulatorState) -> float:
        """
        Calculate reward per step based on the chosen objective.

        :param simulator_state: Current simulator state
        :return: The agent's reward
        """
        succ_ratio, flow_reward = self.get_flow_reward(simulator_state)
        delay, delay_reward = self.get_delay_reward(simulator_state, succ_ratio)

        # weight all objectives as configured before summing them
        flow_reward *= self.flow_weight
        delay_reward *= self.delay_weight

        # calculate and return the sum, ie, total reward
        total_reward = flow_reward + delay_reward
        assert -2 <= total_reward <= 2, f"Unexpected total reward: {total_reward}."

        logger.debug(f"Flow reward: {flow_reward}, success ratio: {succ_ratio}")
        logger.debug(f"Delay reward: {delay_reward}, delay: {delay}")
        logger.debug(f"Total reward: {total_reward}, flow reward: {flow_reward}, delay reward: {delay_reward}")

        return total_reward


if __name__ == '__main__':

    sim = Simulator(network_path, service_path, sim_config_path)

    # get the number of nodes
    network, _, _ = read_network(network_path)
    num_nodes = len(network.nodes)

    # get the number of sfc
    sfc_list = get_sfc(service_path)
    num_sfcs = len(sfc_list)

    # get the number of sf
    max_sf_length = 0
    for _, sf_list in sfc_list.items():
        if max_sf_length < len(sf_list):
            max_sf_length = len(sf_list)
    num_sfs = max_sf_length

    # set flow weight, delay_weight
    flow_weight = 0.5
    delay_weight = 0.5

    action_processor = ActionScheduleProcessor(num_nodes, num_sfcs, num_sfs)
    initial_action = action_processor.process_action(np.random.rand(num_nodes * num_sfcs * num_sfs * num_nodes))

    init_state = sim.init(1234)

    sacoord = CoordProblem(sim, init_state, initial_action, network_path, service_path , flow_weight, delay_weight)

    sacoord.set_schedule(sacoord.auto(minutes=0.2))
    sacoord.copy_strategy = "slice"

    state, e = sacoord.anneal()

    # print(state)
