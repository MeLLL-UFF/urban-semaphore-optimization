from itertools import product

from sympy.utilities.iterables import bracelets


def _convert_bracelet_to_traffic(bracelet, traffic_levels=('light', 'medium', 'heavy')):

    converted_bracelet = tuple([traffic_levels[element] for element in bracelet])

    return converted_bracelet


# it doesn't include mirrored/rotated configurations
def generate_unique_traffic_level_configurations(
        number_of_incoming_streets,
        traffic_levels=('light', 'medium', 'heavy')):

    bracelets_generator = bracelets(number_of_incoming_streets, len(traffic_levels))

    configuration_generator = (_convert_bracelet_to_traffic(bracelet) for bracelet in bracelets_generator)

    return configuration_generator


def generate_all_traffic_level_configurations(number_of_incoming_streets, traffic_levels=('light', 'medium', 'heavy')):

    configuration_generator = product(traffic_levels, repeat=number_of_incoming_streets)

    return configuration_generator

