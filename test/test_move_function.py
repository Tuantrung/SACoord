import numpy as np

from sacoord.sa_coord import CoordProblem
from siminterface.simulator import Simulator
from coordsim.reader.reader import read_network, get_sfc
from datetime import datetime
from rlsp.agents.main import get_base_path
from common.common_functionalities import create_input_file
from sacoord.sa_coord import CoordProblem, get_config, copy_input_files
from rlsp.envs.environment_limits import EnvironmentLimits
from sacoord.sa_simulator_wrapper import SASimulatorWrapper
from pathlib import Path
from tqdm import tqdm
from sacoord.shortest_path import get_solution
import pandas as pd

DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PROJECT_ROOT = str(Path(__file__).parent.parent)

algorithm_config = '../res/config/agent/SA/SA_weighted-f1d0.yaml'
network = '../res/networks/abilene/abilene-in2-rand-cap0-2.graphml'
service = '../res/service_functions/abc.yaml'
sim_config = '../res/config/simulator/det-arrival10_det-size001_duration100.yaml'

algorithm_config_path = f'{PROJECT_ROOT}/{algorithm_config}'
network_path = f'{PROJECT_ROOT}/{network}'
service_path = f'{PROJECT_ROOT}/{service}'
sim_config_path = f'{PROJECT_ROOT}/{sim_config}'
seed = 1233

# get the number of nodes and list of ingress nodes
network, ingress_nodes, _ = read_network(network_path)
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

# get the algorithm config
algorithm_config = get_config(algorithm_config_path)

# get the flow weight and the delay_weight
flow_weight = algorithm_config['flow_weight']
delay_weight = algorithm_config['delay_weight']

# get the base path
base_path = get_base_path(algorithm_config_path, network_path, service_path, sim_config_path)

# set the result dir
results_dir = f"{PROJECT_ROOT}/results/{base_path}/{DATETIME}_seed{seed}"

# create the simulator
sim = Simulator(network_path, service_path, sim_config_path)

initial_action = get_solution(network, service_path)

sacoord_instance = CoordProblem(sim, initial_action, seed, network_path, service_path, flow_weight, delay_weight, results_dir)
sacoord_instance.copy_strategy = "deepcopy"

# schedule = sacoord_instance.auto(minutes=0.5)
sacoord_instance.Tmax = 50000
sacoord_instance.steps = 100000
sacoord_instance.Tmin = 0.23

optimistic_action, e, recorded_best_state = sacoord_instance.anneal()
print("\n")

# df = pd.read_csv(recorded_best_state)

test_sim = Simulator(network_path, service_path, sim_config_path, test_mode=True, test_dir=results_dir)
test_env_limits = EnvironmentLimits(num_nodes, sfc_list, observation_space_len=1)
simulator_wrapper = SASimulatorWrapper(test_sim, test_env_limits)
simulator_wrapper.init(seed)
# simulator_wrapper.apply(initial_action)
for i in tqdm(range(500)):
    simulator_wrapper.apply(optimistic_action)

# for index, _ in tqdm(df.iterrows()):
#     action = df.iloc[index].to_numpy()
#     action = np.reshape(action, 75)
#     simulator_wrapper.apply(action)

# for i in tqdm(range(10)):
#     sacoord_instance.move()

copy_input_files(results_dir, algorithm_config_path, network_path, service_path, sim_config_path)
create_input_file(results_dir, len(ingress_nodes), "SA")
