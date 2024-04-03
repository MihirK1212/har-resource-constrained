import torch.utils.data as torch_data_utils
from abc import ABC, abstractmethod
import torch
import numpy as np
import random
import math

import config

class CustomDataset(torch_data_utils.Dataset):
      def __init__(self, data, target):
          self.data = data
          self.target = target

      def __getitem__(self, index):
          x = self.data[index]
          y = self.target[index]
          return x, y

      def __len__(self):
          return len(self.data)
      
class EnsembleMemberModel(ABC):

    @abstractmethod
    def init_model(self, X_train, X_test, y_train, y_test, num_classes):
      pass

    @abstractmethod
    def fit(self):
      pass

    @abstractmethod
    def get_predictions(self):
      pass  

def get_device():
    return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def horizontal_flip(sequence):
    return np.flip(sequence, axis=1)

def random_scaling(sequence, scale_range=(0.8, 1.2)):
    scale_factor = np.random.uniform(*scale_range)
    scaled_frames = sequence * scale_factor
    return scaled_frames

def shear(sequence, shear_factor=0.2):
    num_frames, num_features = sequence.shape[0], sequence.shape[1]
    assert num_features == 60
    shear_offset = int(shear_factor * num_features)
    for i in range(num_frames):
        shift = np.random.randint(-shear_offset, shear_offset + 1)
        sequence[i] = np.roll(sequence[i], shift)
    return sequence

def get_augmented_sequences(sequence):
    num_frames = sequence.shape[0]
    return [
        horizontal_flip(sequence.reshape(num_frames, -1)).reshape(num_frames, 20, 3),
        random_scaling(sequence.reshape(num_frames, -1)).reshape(num_frames, 20, 3),
        shear(sequence.reshape(num_frames, -1)).reshape(num_frames, 20, 3)
    ]

tssi_order = [
    3, 2, 19, 2, 1, 8, 10, 12, 10, 8, 1, 2,
    0, 7, 9, 11, 9, 7, 0, 2, 3, 6, 4, 14, 16,
    18, 16, 14, 4, 6, 5, 13, 15, 17, 15, 13,
    5, 6, 3, 2
]

def remove_nan(vector):
    vector[np.isnan(vector)] =  np.nanmean(vector)
    return vector

def shuffle(l1, l2):
    combined_data = list(zip(l1, l2))
    random.Random(config.SEED).shuffle(combined_data)
    l1, l2 = zip(*combined_data)
    return l1, l2

def get_tssi_dim_sequence(dim_sequence):
    tssi_dim_sequence = []
    for frame_ind in range(dim_sequence.shape[0]):
        frame = []
        for joint in tssi_order:
            frame.append(dim_sequence[frame_ind][joint])
        tssi_dim_sequence.append(frame)
    return tssi_dim_sequence

def get_tssi_angle_sequence(sequence):
    sequence = torch.tensor(sequence.copy(), dtype=torch.float64)
    sequence = sequence.view(sequence.shape[0], -1)

    def get_angle(point1: torch.Tensor, point2: torch.Tensor) -> float:
      dot_product = torch.dot(point1, point2)
      magnitude1 = torch.norm(point1)
      magnitude2 = torch.norm(point2)
      if magnitude1 == 0 or magnitude2 == 0:
          return 0
      cosine_angle = dot_product / (magnitude1 * magnitude2)
      angle_radians = torch.acos(cosine_angle)
      angle_degrees = math.degrees(angle_radians.item())
      angle_degrees/=360.0
      if abs(angle_degrees) < 1e-8:
        return 0
      if math.isnan(angle_degrees):
          return 0
      return angle_degrees

    def get_augmented_frame(frame: torch.Tensor) -> list:
      angles = []
      for i in tssi_order:
          joint1, joint2 = i, (i+1)%config.NUM_JOINTS
          point1 = frame[3*joint1 : (3*joint1 + 3)]
          point2 = frame[3*joint2 : (3*joint2 + 3)]
          angles.append(get_angle(point1=point1, point2=point2))
      return angles

    tssi_angle_sequence = []
    for frame_ind in range(sequence.shape[0]):
      tssi_angle_sequence.append(get_augmented_frame(frame=sequence[frame_ind]))
    return tssi_angle_sequence