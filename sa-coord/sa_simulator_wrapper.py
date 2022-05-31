import logging
from rlsp.envs.simulator_wrapper import SimulatorWrapper
import numpy as np
from typing import Tuple
from spinterface import SimulatorAction, SimulatorState
from rlsp.envs.action_norm_processor import ActionScheduleProcessor

logger = logging.getLogger(__name__)


class SASimulatorWrapper(SimulatorWrapper):
    def __init__(self, simulator, env_limits):
        super(SASimulatorWrapper, self).__init__(simulator, env_limits)
        self.observations_space = None

    def init(self, seed) -> SimulatorState:
        """Creates a new simulation environment.

        Reuses network_file, service_functions_file from object scope.
        Creates mapping from string identifier to integer IDs for nddes, SFC, and sf
        Calculates shortest paths array for network graph.

        Parameters
        ----------
        seed : int
            The seed value enables reproducible gym environments respectively
            reproducible simulator environments. This value should initialize
            the random number generator used by the simulator when executing
            randomized functions.

        Returns
        -------
        vectorized_state: np.ndarray

        state: SimulatorState
        """
        logger.debug("INIT Simulator")
        # get initial state
        init_state = self.simulator.init(seed)

        # create a mapping such that every node, SF and SFC has a fixed array position
        # this is important for the RL
        # create also an inverted mapping
        node_index = 0
        sfc_index = 0
        sf_index = 0

        for node in init_state.network['nodes']:
            self.node_map[node['id']] = node_index
            node_index = node_index + 1

        self.sfc_dict = init_state.sfcs

        for sfc in init_state.sfcs:
            self.sfc_map[sfc] = sfc_index
            sfc_index = sfc_index + 1

        for service_function in init_state.service_functions:
            self.sf_map[service_function] = sf_index
            sf_index = sf_index + 1

        return init_state

    def apply(self, action_array: np.ndarray) -> SimulatorState:
        """
        Encapsulates the simulators apply method to use the gym interface

        Creates a SimulatorAction object from the agent's return array.
        Applies it to the simulator, translates the returning SimulatorState to an array and returns it.

        Parameters
        ----------
        action_array: np.ndarray

        Returns
        -------
        vectorized_state: dict
        state: SimulatorState
        """
        logger.debug(f"Action array (NN output + noise, normalized): {action_array}")
        action_processor = ActionScheduleProcessor(self.env_limits.MAX_NODE_COUNT, self.env_limits.MAX_SF_CHAIN_COUNT,
                                                   self.env_limits.MAX_SERVICE_FUNCTION_COUNT)
        action_array = action_processor.process_action(action_array)
        scheduling = np.reshape(action_array, self.env_limits.scheduling_shape)

        # initialize with empty schedule and placement for each node, SFC, SF
        scheduling_dict = {v: {sfc: {sf: {} for sf in self.sf_map.keys()} for sfc in self.sfc_map.keys()}
                           for v in self.node_map.keys()}
        placement_dict = {v: set() for v in self.node_map.keys()}

        # parse schedule and prepare dict
        for src_node, src_node_idx in self.node_map.items():
            for sfc, sfc_idx in self.sfc_map.items():
                for sf, sf_idx in self.sf_map.items():
                    for dst_node, dst_node_idx in self.node_map.items():
                        index = (src_node_idx, sfc_idx, sf_idx, dst_node_idx)
                        scheduling_dict[src_node][sfc][sf][dst_node] = scheduling[index]

        # compute dynamic placement depending on schedule and traffic for over all active ingress nodes and all sfcs
        for sfc, sfc_idx in self.sfc_map.items():
            active_ing_nodes = self.simulator.get_active_ingress_nodes()
            logger.debug(f"Active ingress nodes: {active_ing_nodes}")
            for ing in active_ing_nodes:
                # follow possible traffic to calculate sufficient placement
                self.add_placement_recursive(ing, 0, sfc, scheduling_dict, placement_dict)

        # invoke simulator
        logger.debug("call apply on Simulator")
        simulator_action = SimulatorAction(placement_dict, scheduling_dict)
        state = self.simulator.apply(simulator_action)

        return state
