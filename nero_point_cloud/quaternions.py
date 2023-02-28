# rotation-related functions using quaternions
import numpy as np


def axis_angle_to_quaternion(axis, theta, unit='degree'):
    '''
    Convert axis-angle representation to quaternion
    Inputs:
        axis: numpy array, 3-vector as the rotation axis
        theta: rotation angle, in degree or radian
        unit: indicate is theta is in degree or radian
    Returns:
        Quaternion components in scalar-first format [q_0, q_1, q_2, q_3]
    '''

    # convert theta to radian
    if unit == 'degree':
        theta = theta / 180.0 * np.pi

    # make sure that axis is unit-vector
    axis = axis / np.linalg.norm(axis)

    # quaternion in scaler-first format
    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = np.sin(theta / 2) * axis / np.linalg.norm(axis)

    return q


def quaternion_rotate(in_vector, q):
    '''
    Rotate single or a list of 3-vector by a quaternion.
    Inputs:
        in_vector: a single or list of 3-vectors with shape (N, 3) or (3,)
        q: quaternion in scaler-first format
    Returns:
        out_vector: a single or list of 3-vectors with shape (N, 3) or (3,)
    '''

    # single input vector
    if len(in_vector.shape) == 1:
        # convert in_vector into quaternion
        in_vector_quat = np.insert(in_vector, 0, 0, axis=0)
        out_vector = quaternion_multiply([q, in_vector_quat, quaternion_conjugate(q)])[1:]
    # a list of input vectors
    elif len(in_vector.shape) == 2:
        out_vector = np.zeros(in_vector.shape)
        # convert in_vector into quaternion
        in_vector_quat = np.insert(in_vector, 0, 0, axis=1)

        for i, cur_vector_quat in enumerate(in_vector_quat):
            out_vector[i] = quaternion_multiply([q, cur_vector_quat, quaternion_conjugate(q)])[1:]

    return out_vector


def quaternion_conjugate(q):
    '''
    Computes the quaternion conjugate.
    Input:
        q: a single quaternion in scaler-first format
    Returns:
        q_conj: single quaternion that is the conjugate of q
    '''
    q_conj = np.zeros(4)
    q_conj[0] = q[0]
    q_conj[1] = -q[1]
    q_conj[2] = -q[2]
    q_conj[3] = -q[3]

    return q_conj


def quaternion_multiply(quaternions):
    '''
    Multiply a list of quaternions.
    Inputs:
        quaternions: A list of quaternions in scaler-first format, has shape (N, 4)
    Returns:
        q: A single quaternion result, has shape (4,)
    '''

    # initialize result
    res = np.zeros(4)
    q1 = quaternions[0]

    # multiply all the input quaternions
    for q2 in quaternions[1:]:
        # scaler first format
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        # scaler
        res[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2
        # i
        res[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2
        # j
        res[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2
        # k
        res[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2

        # update basis
        q1 = res

    return res


# # test case
# # counter-clockwise rotation of 90 degrees about the z-axis
# axis = [0, 0, 1]
# q = axis_angle_to_quaternion(axis, 90)
# # print(q)
# points = np.array([[1, 0, 0], [0, 1, 0]])
# rotated_points = quaternion_rotate(points, q)
# print(rotated_points)