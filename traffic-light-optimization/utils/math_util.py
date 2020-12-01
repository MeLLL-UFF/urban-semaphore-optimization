
import math

from shapely.geometry import LineString, Point


def line(p1, angle, distance):

    if angle > 2*math.pi:
        angle -= 2*math.pi
    elif angle < 0:
        angle += 2*math.pi

    slope = math.tan(angle)

    # dy/dx = slope
    if slope == 0:
        dx = distance
        dy = 0
    elif math.isinf(slope):
        dx = 0
        dy = distance
    else:
        dx = math.sqrt(distance**2/(slope**2 + 1))
        dy = slope*dx

    if math.pi/2 < angle <= 3*math.pi/2:
        p2 = [
            p1[0] - dx,
            p1[1] - dy
        ]

    elif (0 <= angle <= math.pi / 2) or (3 * math.pi / 2 <= angle <= 2 * math.pi):
        p2 = [
            p1[0] + dx,
            p1[1] + dy
        ]

    return p1, p2


def retrieve_remaining_path(position, polyline_path):

    position_point = Point(position)

    closer_segment_distance = float('inf')
    closer_segment_index = None
    p1 = polyline_path[0]
    for i in range(1, len(polyline_path)):

        p2 = polyline_path[i]

        p1_p2_segment = LineString([p1, p2])

        distance = p1_p2_segment.distance(position_point)

        if distance < closer_segment_distance:
            closer_segment_distance = distance
            closer_segment_index = i

        p1 = p2

    new_polyline = [list(position)]

    for i in range(closer_segment_index, len(polyline_path)):
        p2 = polyline_path[i]
        new_polyline.append(p2)

    return LineString(new_polyline)


def translate_polyline(polyline, x=0, y=0):

    polyline_points = polyline.split()

    translated_polyline = []
    for point in polyline_points:
        coordinates = point.split(',')
        coordinates[0] = str(float(coordinates[0]) + x)
        coordinates[1] = str(float(coordinates[1]) + y)
        translated_polyline.append(coordinates[0] + ',' + coordinates[1])

    return ' '.join(translated_polyline)
