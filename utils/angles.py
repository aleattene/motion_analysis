import numpy as np


# Function to calculate the angle
def calculate_angle_degrees(a, b, c):
    a = np.array(a)  # Start point
    b = np.array(b)  # Intermediate point
    c = np.array(c)  # End point

    # Vectors
    ba = a - b
    bc = c - b

    # Scalar product and vector norm
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Calculate the angle between the two vectors (in degrees)
    cos_angle = dot_product / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    return angle


def calculate_angle_reverse_degrees(a, b, c):
    a = np.array(a)  # Start point
    b = np.array(b)  # Intermediate point
    c = np.array(c)  # End point

    # Vectors
    ba = a - b
    bc = c - b

    # Scalar product and vector norm
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Calculate the angle between the two vectors (in degrees)
    cos_angle = dot_product / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Vector product to determine the direction of the angle
    # cross_product = np.cross(ba, bc)

    return 360 - angle


# def calculate_angle_rotation(shoulder_center, hip_center):
#     """
#     Calculate the angle of rotation of the body with respect to the initial position.
#     :param shoulder_center: shoulder center [x, y]
#     :param hip_center: hip center [x, y]
#     :return: angle of rotation in degrees
#     """
#     delta_x = shoulder_center[0] - hip_center[0]
#     delta_y = shoulder_center[1] - hip_center[1]
#
#     # Angle relative to the vertical axis
#     angle_rad = np.arctan2(delta_y, delta_x)
#     angle_deg = np.degrees(angle_rad)
#
#     # Angle Normalization
#     if angle_deg < 0:
#         angle_deg += 360
#
#     # Set the initial position as the vertical axis
#     angle_from_vertical = (angle_deg - 90) % 360
#
#     # Angle conversion to the range [0, 180]
#     if angle_from_vertical > 180:
#         angle_from_vertical = 360 - angle_from_vertical
#
#     return angle_from_vertical


def calculate_rotation_angle(initial_shoulder, initial_hand, current_shoulder, current_hand):
    """
    Calculate the rotation angle of the right arm with respect to the initial position.
    :param initial_shoulder: initial coordinate of the right shoulder [x, y]
    :param initial_hand: initial coordinate of the right hand [x, y]
    :param current_shoulder: current coordinate of the right shoulder [x, y]
    :param current_hand: current coordinate of the right hand [x, y]
    :return: rotation angle in degrees
    """

    # Initial vector
    initial_vector = np.array(initial_hand) - np.array(initial_shoulder)
    initial_angle = np.arctan2(initial_vector[1], initial_vector[0])

    # Current vector
    current_vector = np.array(current_hand) - np.array(current_shoulder)
    current_angle = np.arctan2(current_vector[1], current_vector[0])

    # Calculate the rotation angle
    rotation_angle_rad = current_angle - initial_angle
    rotation_angle_deg = np.degrees(rotation_angle_rad)

    # Angle normalization
    if rotation_angle_deg < -180:
        rotation_angle_deg += 360
    elif rotation_angle_deg > 180:
        rotation_angle_deg -= 360

    return abs(rotation_angle_deg)  # Return the absolute value of the angle
