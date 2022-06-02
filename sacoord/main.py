import logging
from siminterface.simulator import Simulator
from coordsim.reader.reader import read_network, get_sfc
from rlsp.envs.action_norm_processor import ActionScheduleProcessor
from datetime import datetime
from rlsp.agents.main import get_base_path
from common.common_functionalities import create_input_file
import numpy as np
import click
from sacoord.sa_coord import CoordProblem, get_config, copy_input_files
import random

sigma = 0.2
mu = 0.0
# _defaults_algorithm_config_path = '../res/config/agent/SA/SA_weighted-f1d0.yaml'
# _defaults_network_path = '../res/networks/5node/5node-in2-rand-cap0-2.graphml'
# _defaults_service_path = '../res/service_functions/abc.yaml'
# _defaults_sim_config_path = '../res/config/simulator/det-arrival10_det-size001_duration100.yaml'
# _defaults_seed = 1234

DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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

    algorithm_config_path = algorithm_config
    network_path = network
    service_path = service
    sim_config_path = sim_config
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
    results_dir = f"../results/{base_path}/{DATETIME}_seed{seed}"

    # create the simulator
    sim = Simulator(network_path, service_path, sim_config_path, test_mode=True, test_dir=results_dir)

    # initial action
    action_processor = ActionScheduleProcessor(num_nodes, num_sfcs, num_sfs)
    equal_action = np.zeros(num_nodes * num_sfcs * num_sfs * num_nodes)
    initial_action = action_processor.process_action(equal_action)

    sacoord_instance = CoordProblem(sim, initial_action, seed, network_path, service_path, flow_weight, delay_weight)
    sacoord_instance.copy_strategy = "slice"

    try:
        sacoord_instance.steps = algorithm_config['steps']
    except:
        logging.info("The steps is not setting, continue with the defaults value")

    # sacoord.set_schedule(sacoord.auto(minutes=0.05))

    optimistic_action, e = sacoord_instance.anneal()
    assert len(optimistic_action) == len(initial_action)

    # copy input file to the results directory
    copy_input_files(results_dir, algorithm_config_path, network_path, service_path, sim_config_path)

    # create information algorithms file
    create_input_file(results_dir, len(ingress_nodes), "SA")


if __name__ == '__main__':
    _defaults_algorithm_config = '../res/config/agent/SA/SA_weighted-f1d0.yaml'
    _defaults_network = '../res/networks/5node/5node-in2-rand-cap0-2.graphml'
    _defaults_service = '../res/service_functions/abc.yaml'
    _defaults_sim_config = '../res/config/simulator/det-arrival10_det-size001_duration100.yaml'
    _defaults_seed = '1234'
    cli([_defaults_algorithm_config, _defaults_network, _defaults_service, _defaults_sim_config, '--seed', _defaults_seed])
