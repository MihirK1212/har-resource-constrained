import numpy as np
import copy 

import utils
import config

def skelemotion_aggregation(sequence):

    agg_sequence = copy.deepcopy(sequence)
    num_frames, d = agg_sequence.shape[0], 5

    dim_sequence_x, dim_sequence_y, dim_sequence_z = agg_sequence[:, :, 0], agg_sequence[:, :, 1], agg_sequence[:, :, 2]
    dim_sequence_x, dim_sequence_y, dim_sequence_z = utils.get_tssi_dim_sequence(dim_sequence_x), utils.get_tssi_dim_sequence(dim_sequence_y), utils.get_tssi_dim_sequence(dim_sequence_z)

    num_joints = len(config.tssi_order)

    agg_sequence = np.stack([dim_sequence_x, dim_sequence_y, dim_sequence_z], axis=2)

    D = np.zeros(((num_frames-d), num_joints, 3))
    for t in range(num_frames-d):
        for c in config.tssi_order:
            D[t][c] = agg_sequence[t+d][c] - agg_sequence[t][c]

    M = np.sqrt(np.sum(D**2, axis=2))
    theta_xy = np.arctan2(D[:, :, 0], D[:, :, 1])
    theta_yz = np.arctan2(D[:, :, 2], D[:, :, 1])
    theta_zx = np.arctan2(D[:, :, 0], D[:, :, 2])

    res = np.concatenate([M, theta_xy, theta_yz, theta_zx], axis = 1)

    f = np.mean(res, axis=0)
    return np.array(f)