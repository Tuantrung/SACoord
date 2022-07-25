from siminterface.simulator import Simulator
from coordsim.reader.reader import read_network, get_sfc
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from datetime import datetime
from rlsp.agents.main import get_base_path
from common.common_functionalities import create_input_file
from sacoord.sa_coord import CoordProblem, get_config, copy_input_files
from rlsp.envs.environment_limits import EnvironmentLimits
from sacoord.sa_simulator_wrapper import SASimulatorWrapper
import numpy as np
import click
import random
import logging
from pathlib import Path
from sacoord.greedy import GreedyCoord
from sacoord.shortest_path import get_solution

# sigma = 0.2
# mu = 0.0

DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
PROJECT_ROOT = str(Path(__file__).parent.parent)
logger = logging.getLogger(__name__)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('algorithm_config', type=click.Path(exists=True))
@click.argument('network', type=click.Path(exists=True))
@click.argument('service', type=click.Path(exists=True))
@click.argument('sim_config', type=click.Path(exists=True))
@click.option('--seed', default=random.randint(1000, 9999),
              help="Specify the random seed for the environment and the learning agent.")
# @click.option('-v', '--verbose', is_flag=True, help="Set console logger level to debug. (Default is INFO)")
def cli(algorithm_config, network, service, sim_config, seed):
    """sa cli for running"""
    global logger

    algorithm_config_path = f'{PROJECT_ROOT}/{algorithm_config}'
    network_path = f'{PROJECT_ROOT}/{network}'
    service_path = f'{PROJECT_ROOT}/{service}'
    sim_config_path = f'{PROJECT_ROOT}/{sim_config}'
    seed = seed

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

    # initial action load balance
    # action_processor = ActionScheduleProcessor(num_nodes, num_sfcs, num_sfs)
    # equal_action = np.zeros(num_nodes * num_sfcs * num_sfs * num_nodes)
    # initial_action = action_processor.process_action(equal_action)

    # initial_action = np.zeros(num_nodes * num_sfcs * num_sfs * num_nodes)
    # pop0_schedule = np.array([0, 0.5, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0.48, 0.26, 0, 0.26])
    # greedy = GreedyCoord(network, ingress_nodes, num_sfcs, num_sfs, flow_weight, delay_weight)
    # pop0_schedule = np.array([0, 0.5, 0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0, 0.5, 0, 0.5])
    # pop0_schedule = greedy.greedy_schedule()
    # for i in range(num_nodes):
    #     t = i * (num_nodes * num_sfs)
    #     initial_action[t:t + num_nodes * num_sfs] = initial_action[t:t + num_nodes * num_sfs] + pop0_schedule

    # get initial action from shortest path solution
    initial_action = get_solution(network, service_path)

    sacoord_instance = CoordProblem(sim, initial_action, seed, network_path, service_path, flow_weight, delay_weight, results_dir)
    sacoord_instance.copy_strategy = "slice"

    # try:
    #     sacoord_instance.steps = algorithm_config['steps']
    # except:
    #     logging.info("The steps is not setting, continue with the defaults value")

    schedule = sacoord_instance.auto(minutes=0.5)
    sacoord_instance.set_schedule(schedule)

    # test best state record
    test_sim = Simulator(network_path, service_path, sim_config_path, test_mode=True, test_dir=results_dir)
    test_env_limits = EnvironmentLimits(num_nodes, sfc_list, 1)
    simulator_wrapper = SASimulatorWrapper(test_sim, test_env_limits)
    simulator_wrapper.init(seed)
    for i in range(sacoord_instance.steps):
        simulator_wrapper.apply(recorded_best_state[i])

    # copy input file to the results directory
    copy_input_files(results_dir, algorithm_config_path, network_path, service_path, sim_config_path)

    # create information algorithms file
    create_input_file(results_dir, len(ingress_nodes), "SA")
    print("\n")


if __name__ == '__main__':
    _defaults_algorithm_config = 'res/config/agent/SA/SA_weighted-f1d0.yaml'
    _defaults_network = 'res/networks/abilene/abilene-in1-rand-cap0-2.graphml'
    _defaults_service = 'res/service_functions/abc.yaml'
    _defaults_sim_config = 'res/config/simulator/' \
                           'rand-arrival10_det-size001_duration100_trace_0_100_inall_chrate2_scale00005.yaml'
    _defaults_seed = '1234'
    cli([_defaults_algorithm_config, _defaults_network, _defaults_service, _defaults_sim_config, '--seed', _defaults_seed])
