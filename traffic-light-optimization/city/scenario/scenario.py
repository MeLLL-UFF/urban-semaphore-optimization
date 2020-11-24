
import math

from utils.bidict import bidict


class Scenario:

    def __init__(self, name='', flows_to_direction=None, _traffic_parametrization=None,
                 _traffic_distribution=None):

        if flows_to_direction is None:
            flows_to_direction = bidict()

        if _traffic_parametrization is None:
            _traffic_parametrization = {}

        if _traffic_distribution is None:
            _traffic_distribution = {}

        self.name = name
        self.flows_to_direction = flows_to_direction

        self._traffic_parametrization = _traffic_parametrization
        self._traffic_distribution = _traffic_distribution

        self.traffic_configuration = {}
        self._traffic_configuration_generator_object = None

        self.cars_total = 0


    def __iter__(self):
        self._traffic_configuration_generator_object = self._init_traffic_configuration_generator()
        return self

    def __next__(self):
        self.traffic_configuration = next(self._traffic_configuration_generator_object)
        return self.traffic_configuration

    def __repr__(self):
        repr = self.name + ': '

        directions_repr = []
        for direction, cars_total_current in self.traffic_configuration.items():
            directions_repr.append(direction + ' - ' + str(cars_total_current))

        repr += ' | '.join(directions_repr)

        return repr

    def get_total_of_flow_vehicles(self, flow_id):

        proportion = self._traffic_distribution[flow_id]

        direction = self.flows_to_direction[flow_id]
        total_direction_vehicles = self.traffic_configuration[direction]['cars_total_current']

        total_flow_vehicles = math.ceil(total_direction_vehicles*proportion)

        return total_flow_vehicles

    def create_traffic_configuration(self, *args):

        assert len(args) == len(self.flows_to_direction.inverse.keys())

        traffic_configuration = {}
        for index, direction in enumerate(self.flows_to_direction.inverse.keys()):
            traffic_configuration[direction] = {'cars_total_current': args[index]}

        self.traffic_configuration = traffic_configuration
        return traffic_configuration

    def _init_traffic_configuration_generator(self):

        traffic_configuration = {
           direction: {
               'cars_total_current': 0
           } for direction in self.flows_to_direction.inverse.keys()
        }

        ranges = [self._traffic_parametrization[direction]['cars_total_range']
                  for direction in self.flows_to_direction.inverse.keys()]

        generators = [iter(range(r['start'], r['stop'] + 1, r['step'])) for r in ranges]
        directions = list(traffic_configuration.keys())

        # Generating a customizable-depth nested 'for' loop
        #
        # generate next number                                  (Iterate current 'for' loop)
        # did it raise StopIteration?
        #   yes                                                 (The current 'for' iteration ended)
        #       can you go to the left?
        #           yes                                             (It is not the outermost 'for')
        #               restart the generator
        #               go to the left                                  (Go to an outer 'for')
        #               continue
        #           no                                              (It is the outermost 'for')
        #               break                                           (All the 'for' loops ended)
        #   no                                                  ('for' did iterate with success)
        #       can you go to the right?
        #           yes                                             (It is not the innermost 'for')
        #               go to the right                                 (Go to an inner for)
        #               continue
        #           no                                              (It is the innermost 'for')
        #               yield generated numbers                         (Provide scenario configuration)
        #               continue

        i = 0
        done_looping = False
        while not done_looping:
            try:
                direction = directions[i]
                traffic_configuration[direction]['cars_total_current'] = next(generators[i])
            except StopIteration:
                if i > 0:
                    generators[i] = iter(range(ranges[i]['start'], ranges[i]['stop'] + 1, ranges[i]['step']))
                    i -= 1
                else:
                    done_looping = True
            else:
                if i < len(ranges) - 1:
                    i += 1
                else:
                    self.traffic_configuration = traffic_configuration
                    self.cars_total = sum([self.get_total_of_flow_vehicles(flow_id)
                                           for flow_id in self.flows_to_direction.keys()])
                    yield traffic_configuration
