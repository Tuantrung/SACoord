import logging
from simanneal import Annealer
from rl.random import GaussianWhiteNoiseProcess
from coordsim.reader.reader import read_network, get_sfc, get_sf, network_diameter
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from sacoord.sa_simulator_wrapper import SASimulatorWrapper
from datetime import datetime
from rlsp.envs.environment_limits import EnvironmentLimits
import numpy as np
from spinterface import SimulatorState
import yaml
import os
from shutil import copyfile


logger = logging.getLogger(__name__)

# seed = 1234
sigma = 0.2
mu = 0.0
# algorithm_config_path = '../res/config/agent/SA/SA_weighted-f1d0.yaml'
# network_path = '../res/networks/5node/5node-in2-rand-cap0-2.graphml'
# service_path = '../res/service_functions/abc.yaml'
# sim_config_path = '../res/config/simulator/det-arrival10_det-size001_duration100.yaml'


class CoordProblem(Annealer):
    """
    Implement annealer with Coord Problem.
    """
    simulator_state: SimulatorState

    # pass extra data into the constructor
    def __init__(self, simulator, action, seed, network_file, service_file, flow_weight, delay_weight):
        self.simulator = simulator
        self.action = action
        self.seed = seed

        self.network, _, _ = read_network(network_file)
        self.num_nodes = len(self.network)
        self.network_diameter = network_diameter(self.network)

        self.sfc_list = get_sfc(service_file)
        self.num_sfcs = len(self.sfc_list)

        self.sf_list = get_sf(service_file)
        self.num_sfs = len(self.sf_list)

        self.tmp = self.num_nodes * self.num_sfcs * self.num_sfs

        self.action_processor = ActionScheduleProcessor(self.num_nodes, self.num_sfcs, self.num_sfs)
        self.env_limits = EnvironmentLimits(len(self.network.nodes), self.sfc_list, 1)
        self.simulator_wrapper = SASimulatorWrapper(self.simulator, self.env_limits)
        self.simulator_wrapper.init(self.seed)
        self.curr_simulator_state = self.simulator_wrapper.apply(action)

        self.flow_weight = flow_weight
        self.delay_weight = delay_weight

        self.min_delay, self.max_delay = self.min_max_delay()
        super().__init__(initial_state=self.action)

    def move(self):
        """Add Gaussian White Noise to action"""
        initial_energy = self.energy()

        random_process = GaussianWhiteNoiseProcess(sigma=sigma, mu=mu, size=self.tmp)
        noise = random_process.sample()

        for i in range(self.num_nodes):
            t = i * self.num_nodes
            self.action[t: t + self.tmp] = self.action[t: t + self.tmp] + noise

        self.curr_simulator_state = self.simulator_wrapper.apply(self.action)

        # print(self.curr_simulator_state.placement)
        # print(self.curr_simulator_state.traffic['pop0']['sfc_1'])
        # print(self.curr_simulator_state.traffic['pop1']['sfc_1'])
        # print(self.curr_simulator_state.traffic['pop2']['sfc_1'])
        # print(self.curr_simulator_state.traffic['pop3']['sfc_1'])
        # print(self.curr_simulator_state.traffic['pop4']['sfc_1'])
        # print("\n")

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
            # flow_reward = (cur_drop_flow - cur_succ_flow) / (cur_succ_flow + cur_drop_flow)
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
            # delay_reward = (delay - self.min_delay) / self.network_diameter
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


def get_config(config_file):
    """Parse agent config params in specified yaml file and return as Python dict"""
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def copy_input_files(target_dir, algorithm_config_path, network_path, service_path, sim_config_path):
    """Create the results directory and copy input files"""
    new_algorithm_config_path = f"{target_dir}/{os.path.basename(algorithm_config_path)}"
    new_network_path = f"{target_dir}/{os.path.basename(network_path)}"
    new_service_path = f"{target_dir}/{os.path.basename(service_path)}"
    new_sim_config_path = f"{target_dir}/{os.path.basename(sim_config_path)}"

    os.makedirs(target_dir, exist_ok=True)
    copyfile(algorithm_config_path, new_algorithm_config_path)
    copyfile(network_path, new_network_path)
    copyfile(service_path, new_service_path)
    copyfile(sim_config_path, new_sim_config_path)
