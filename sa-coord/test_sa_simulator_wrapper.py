from sa_simulator_wrapper import SASimulatorWrapper
from siminterface import Simulator
from rl.random import GaussianWhiteNoiseProcess
import numpy as np
from rlsp.envs.environment_limits import EnvironmentLimits
from coordsim.reader.reader import read_network, get_sfc
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from datetime import datetime
from pathlib import Path
from rlsp.agents.main import get_base_path
from common.common_functionalities import create_input_file, copy_input_files
import os
from tqdm import tqdm


# initial parameter
network_path = '../params/networks/triangle.graphml'
service_path = '../params/services/abc.yaml'
sim_config_path = '../params/config/sim_config.yaml'
sigma = 0.2
mu = 0.0
seed = 3456

DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# get base path
base_path = get_base_path('', network_path, service_path, sim_config_path)

# set result dir
results_dir = f"../results/{base_path}/{DATETIME}_seed{seed}"

# get the number of nodes
network, ingress_nodes, _ = read_network(network_path)

# get the number of sfc
sfc_list = get_sfc(service_path)

# get the number of sf
max_sf_length = 0
for _, sf_list in sfc_list.items():
    if max_sf_length < len(sf_list):
        max_sf_length = len(sf_list)
num_sfs = max_sf_length

# define environment limits
env_limits = EnvironmentLimits(len(network.nodes), sfc_list, 1)

# initialize simulator
simulator = Simulator(network_path, service_path, sim_config_path, test_mode=True, test_dir=results_dir)
sa_simulator_wrapper = SASimulatorWrapper(simulator, env_limits)

# set up simulator wrapper state
initial_state = sa_simulator_wrapper.init(seed)
action_processor = ActionScheduleProcessor(env_limits.MAX_NODE_COUNT, env_limits.MAX_SF_CHAIN_COUNT,
                                           env_limits.MAX_SERVICE_FUNCTION_COUNT)
action_array = np.zeros(len(network.nodes) * num_sfs * num_sfs)
action = action_processor.process_action(action_array)

for i in tqdm(range(3)):
    # add noise
    random_process = GaussianWhiteNoiseProcess(sigma=sigma, mu=mu, size=27)
    noise = random_process.sample()
    action_array = action_array + noise

    action = action_processor.process_action(action_array)
    state = sa_simulator_wrapper.apply(action)

# copy the input files(network, simulator config....) to  the results directory
copy_input_files(results_dir, os.path.abspath(network_path), os.path.abspath(service_path),
                 os.path.abspath(sim_config_path))

create_input_file(results_dir, len(ingress_nodes), "SA")