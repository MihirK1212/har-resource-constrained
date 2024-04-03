import copy
import numpy as np

from algorithms import statistical_moments_aggregation
import config

def get_sequence_windows(sequence, required_windows = config.NUM_WINDOWS, min_stride = 2):
    sequence = copy.deepcopy(sequence)
    num_frames = sequence.shape[0]
    stride = max(min_stride, num_frames//required_windows)
    window_size = num_frames - (required_windows - 1)*stride
    assert window_size >= stride
    lb, ub = 0, window_size - 1
    windows = []
    while (lb<=ub and ub<num_frames):
      window = sequence[lb:(ub+1)]
      windows.append(window)
      lb+=stride
      ub+=stride
    assert ub == (num_frames - 1 + stride)
    assert len(windows) == required_windows
    for window in windows:
      assert window.shape[0] == window_size
    return windows
    
def window_division_aggregation(sequence):
    agg_sequence = copy.deepcopy(sequence)
    windows = get_sequence_windows(sequence=agg_sequence)
    res = []
    for window in windows:
      window_mean_feature_vector = statistical_moments_aggregation.statistical_moments_aggregation(sequence=window)
      res.append(window_mean_feature_vector)
    if config.WINDOW_DIVISION_COMBINE_USING_MEAN:
      return np.mean(np.array(res), axis=0)
    else:
      return np.concatenate(res)