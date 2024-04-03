import numpy as np
from scipy.stats import skew, kurtosis

import utils
import copy
import config
from algorithms import segment_distance_normalization

def aggregate_dim_sequence_statistical_moments(dim_sequence):

    dim_sequence    = utils.remove_nan(np.array(dim_sequence, dtype=np.float64))
    mean_vector     = utils.remove_nan(np.mean(dim_sequence, axis=0))
    variance_vector = utils.remove_nan(np.var(dim_sequence, axis=0))
    skewness_vector = utils.remove_nan(skew(dim_sequence, axis=0))
    kurtosis_vector = utils.remove_nan(kurtosis(dim_sequence, axis=0))

    moment_representation = np.concatenate((mean_vector, variance_vector, skewness_vector, kurtosis_vector), axis=0)
    moment_representation = utils.remove_nan(moment_representation)

    if np.all(np.isnan(moment_representation)):
        print('skipped')
        raise ValueError

    return moment_representation

def aggregate_angle_sequence_statistical_moments(angle_sequence):

    mean_vector     = utils.remove_nan(np.mean(angle_sequence, axis=0))
    variance_vector = utils.remove_nan(np.var(angle_sequence, axis=0))
    skewness_vector = utils.remove_nan(skew(angle_sequence, axis=0))
    kurtosis_vector = utils.remove_nan(kurtosis(angle_sequence, axis=0))

    moment_representation = np.concatenate((mean_vector, variance_vector, skewness_vector, kurtosis_vector), axis=0)
    moment_representation = utils.remove_nan(moment_representation)

    if np.all(np.isnan(moment_representation)):
        print('skipped')
        raise ValueError

    return moment_representation.tolist()

def statistical_moments_aggregation(sequence):

    agg_sequence = copy.deepcopy(sequence)
    orig_sequence = copy.deepcopy(sequence)

    angle_sequence = utils.get_tssi_angle_sequence(orig_sequence)

    index_to_subtract = config.ROOT_JOINT_INDEX
    for frame_idx in range(agg_sequence.shape[0]):
        vector_to_subtract = agg_sequence[frame_idx, index_to_subtract, :]
        agg_sequence[frame_idx, :, :] -= vector_to_subtract

    if config.APPLY_SEGMENT_DISTANCE_NORMALIZATION:
      agg_sequence = segment_distance_normalization.apply_segment_distance_normalization(agg_sequence)
      segment_distance_normalization.check_normalization(agg_sequence)

    dim_sequence_x, dim_sequence_y, dim_sequence_z = agg_sequence[:, :, 0], agg_sequence[:, :, 1], agg_sequence[:, :, 2]
    dim_sequence_x, dim_sequence_y, dim_sequence_z = utils.get_tssi_dim_sequence(dim_sequence_x), utils.get_tssi_dim_sequence(dim_sequence_y), utils.get_tssi_dim_sequence(dim_sequence_z)

    mean_feature_vector = []

    x_rep, y_rep, z_rep = aggregate_dim_sequence_statistical_moments(dim_sequence_x), aggregate_dim_sequence_statistical_moments(dim_sequence_y), aggregate_dim_sequence_statistical_moments(dim_sequence_z)
    for i in range(len(utils.tssi_order)):
      mean_feature_vector.extend([x_rep[i], x_rep[2*i], x_rep[3*i], x_rep[4*i]])
    for i in range(len(utils.tssi_order)):
      mean_feature_vector.extend([y_rep[i], y_rep[2*i], y_rep[3*i], y_rep[4*i]])
    for i in range(len(utils.tssi_order)):
      mean_feature_vector.extend([z_rep[i], z_rep[2*i], z_rep[3*i], z_rep[4*i]])

    angle_rep = aggregate_angle_sequence_statistical_moments(angle_sequence=angle_sequence)
    for i in range(len(utils.tssi_order)):
     mean_feature_vector.extend([angle_rep[i], angle_rep[2*i], angle_rep[3*i], angle_rep[4*i]])

    return np.array(mean_feature_vector)