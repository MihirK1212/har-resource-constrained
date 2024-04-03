import torch
import copy 
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np

import config
import utils
from algorithms import segment_distance_normalization

def dtw_aggregate(sequence: torch.Tensor) -> torch.Tensor:
      num_frames, frame_dim = sequence.size()
      aggregated_representation = torch.zeros(frame_dim, dtype=sequence.dtype)
      for i in range(num_frames - 1):
          frame1 = sequence[i].numpy()
          frame2 = sequence[i + 1].numpy()
          distance, path = fastdtw(frame1.reshape(-1, 1), frame2.reshape(-1, 1), dist=euclidean)
          aligned_frames = torch.from_numpy(np.array([frame1[j] for _, j in path]))
          aggregated_representation += aligned_frames.mean(dim=0)
      aggregated_representation /= (num_frames - 1)
      return aggregated_representation

def aggregate_dim_sequence_dtw(dim_sequence):
    dim_sequence = torch.tensor(dim_sequence, dtype=torch.float64)
    dtw_representation = dtw_aggregate(dim_sequence)
    return dtw_representation.tolist()

def aggregate_angle_sequence_dtw(angle_sequence):
  angle_sequence = torch.tensor(angle_sequence, dtype=torch.float64)
  dtw_angle_representation = dtw_aggregate(angle_sequence)
  return dtw_angle_representation.tolist()

def dtw_aggregation(sequence):

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

    x_rep, y_rep, z_rep = aggregate_dim_sequence_dtw(dim_sequence_x), aggregate_dim_sequence_dtw(dim_sequence_y), aggregate_dim_sequence_dtw(dim_sequence_z)
    angle_rep = aggregate_angle_sequence_dtw(angle_sequence=angle_sequence)
    mean_feature_vector.extend(x_rep)
    mean_feature_vector.extend(y_rep)
    mean_feature_vector.extend(z_rep)
    mean_feature_vector.extend(angle_rep)
    return np.array(mean_feature_vector)