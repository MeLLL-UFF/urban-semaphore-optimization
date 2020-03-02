import ast
import random
from collections import deque

import numpy as np
import traci

from city.traffic_light_system.traffic_light.traffic_light import TrafficLight
from city.traffic_light_system.traffic_light.reinforcement_learning.deep_q_network import DeepQNetwork


class ReinforcementLearningTrafficLight(TrafficLight):

    def __init__(self, id, parameters, **kwargs):

        super().__init__(id, parameters)

        seed = kwargs.get('seed', None)
        self._experiment = kwargs.get('experiment', None)

        if seed is not None:
            random.seed(seed)

        definition = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        self.phases = definition.phases
        self.optimizable_phases = ast.literal_eval(definition.subParameter['optimizable_phases'])

        self.edges = self._get_edges()
        self.street_names = list(map(lambda x: x.split('#')[0], self.edges))

        self.current_phase = None
        self.action_taken = None

        self.reward_moving = 0
        self.reward_halting = 0

        self.total_time_loss = 0

        self._next_desirable_phase_queue = deque()

        self._update_current_phase()
        self.state = self._get_state()

        self.network = None
        gamma = parameters.get(str(self.id) + '.' + 'gamma', 0.95)
        epsilon = parameters.get(str(self.id) + '.' + 'epsilon', 0.01)
        learning_rate = parameters.get(str(self.id) + '.' + 'learning_rate', 0.00002)
        use_memory_palace = parameters.get(str(self.id) + '.' + 'memory_palace', True)
        batch_size = parameters.get(str(self.id) + '.' + 'batch_size', 32)

        self._build_network(gamma, epsilon, learning_rate, batch_size, use_memory_palace)

    def __repr__(self):
        return 'id: ' + str(self.id) + '; Green phase: ' + self.phases[self.current_phase].name

    def step(self):

        self._update_current_phase()

        already_updated_state = False
        if self.action_taken is not None:

            previous_state = self.state

            reward = self._calculate_reward()

            self.state = self._get_state()
            already_updated_state = True

            self._remember(previous_state, self.action_taken, reward, self.state)

            self.action_taken = None

        if self.current_phase in self.optimizable_phases:
            if not already_updated_state:
                self.state = self._get_state()
            self.action_taken = self._act()

    def stop(self):
        self.network.save()

    def _get_state(self):

        total_number_of_lanes = 8
        max_cell_distance = 8

        position_matrix = np.zeros((total_number_of_lanes, max_cell_distance))
        velocity_matrix = np.zeros((total_number_of_lanes, max_cell_distance))

        cell_length = 7

        lanes_count = 0
        for edge in self.edges:
            edge_vehicles = traci.edge.getLastStepVehicleIDs(edge)

            for vehicle in edge_vehicles:

                next_tls_distance = self._get_next_tls_distance(vehicle)
                vehicle_speed = traci.vehicle.getSpeed(vehicle)

                lane_id = traci.vehicle.getLaneID(vehicle)
                speed_limit = traci.lane.getMaxSpeed(lane_id)

                lane_index = traci.vehicle.getLaneIndex(vehicle)
                index = int(next_tls_distance / cell_length)
                if index < max_cell_distance:
                    position_matrix[lanes_count + lane_index][index] = 1
                    velocity_matrix[lanes_count + lane_index][index] = vehicle_speed / speed_limit

            number_of_lanes = traci.edge.getLaneNumber(edge)
            lanes_count += number_of_lanes

        position = np.array(position_matrix)
        position = position.reshape(1, 8, 8, 1)

        velocity = np.array(velocity_matrix)
        velocity = velocity.reshape(1, 8, 8, 1)

        traffic_light = np.array(self.current_phase)
        traffic_light = traffic_light.reshape(1, 1, 1)

        self.state = [position, velocity, traffic_light]
        return self.state

    def _act(self):

        action = self.network.act(self.state)

        if action < len(self.optimizable_phases):
            preferred_phase = self.optimizable_phases[action]
        else:
            # In cases where there is only one green phase
            preferred_phase = -1

        if self.current_phase != preferred_phase:
            self._set_next_phase()

            if preferred_phase != -1:
                self._next_desirable_phase_queue.append(preferred_phase)

        self._update_time_loss()

        self._update_metrics()

        return action

    def _remember(self, state, action, reward, next_state):

        traffic_light_state = state[2][0][0][0]
        memory_palace_state = self.optimizable_phases.index(traffic_light_state)
        memory_palace_action = action

        self.network.remember(state, action, reward, next_state, False, memory_palace_state, memory_palace_action)

        if self.network.get_memory_size() > self.network.batch_size:
            self.network.replay()

    def _calculate_reward(self):
        return self.reward_moving - self.reward_halting

    def _update_metrics(self):

        green_phase_name = self.phases[self.current_phase].name

        green_phase_edges = []
        red_phase_edges = []
        for index, name in enumerate(self.street_names):

            edge = self.edges[index]
            if name in green_phase_name.split('__'):
                green_phase_edges.append(edge)
            else:
                red_phase_edges.append(edge)

        # Calculates reward using the halting cars in the halted edges and all the cars in the moving edges
        self.reward_moving = max(map(traci.edge.getLastStepVehicleNumber, green_phase_edges), default=0)
        self.reward_halting = max(map(traci.edge.getLastStepVehicleNumber, red_phase_edges), default=0)

    def _get_next_tls_distance(self, vehicle):
        next_tls = traci.vehicle.getNextTLS(vehicle)[0]

        next_tls_id = next_tls[0]
        next_tls_distance = next_tls[2]

        if next_tls_id != self.id:
            return "Incorrect semaphore: The first position from getNextTLS isn't returning the expected TLS"

        return next_tls_distance

    def _update_time_loss(self):

        time_loss = 0
        for edge in self.edges:
            edge_vehicles = traci.edge.getLastStepVehicleIDs(edge)

            for vehicle in edge_vehicles:
                speed_limit = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(vehicle))
                vehicle_speed = traci.vehicle.getSpeed(vehicle)
                time_loss += 1 - vehicle_speed / speed_limit

        self.total_time_loss += time_loss

    def _get_edges(self):
        lanes = traci.trafficlight.getControlledLanes(self.id)
        all_edges = [traci.lane.getEdgeID(lane) for lane in lanes]
        # removing walking areas
        edges = list(filter(lambda x: '#' in x, dict.fromkeys(all_edges).keys()))

        return edges

    def _set_next_phase(self):

        next_phase = self._get_next_phase()

        self.current_phase = next_phase
        traci.trafficlight.setPhase(self.id, self.current_phase)

    def _get_next_phase(self):

        if self.current_phase == len(self.phases) - 1:
            return 0
        else:
            return self.current_phase + 1

    def _update_current_phase(self):

        if self._next_desirable_phase_queue and self._get_next_phase() in self.optimizable_phases:
            remaining_time = traci.trafficlight.getNextSwitch(self.id) - traci.simulation.getTime()
            if remaining_time == 0:
                next_phase = self._next_desirable_phase_queue.popleft()
                traci.trafficlight.setPhase(self.id, next_phase)

        self.current_phase = traci.trafficlight.getPhase(self.id)

    def _build_network(self, gamma, epsilon, learning_rate, batch_size, use_memory_palace):

        actions_total = len(self.optimizable_phases) if len(self.optimizable_phases) > 1 else 2
        use_previous_model = False

        memory_palace_states_total = len(self.optimizable_phases)
        memory_palace_actions_total = actions_total

        model_path = self._experiment.path if self._experiment else '.'
        model_name_suffix = self._experiment.name if self._experiment else ''

        self.network = DeepQNetwork(
            gamma, epsilon, learning_rate, batch_size, use_memory_palace,
            memory_palace_states_total, memory_palace_actions_total,
            actions_total, use_previous_model,
            model_path, model_name_suffix
        )