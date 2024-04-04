from os import listdir
from os.path import join
import scipy.io as sio
import numpy as np
from collections import Counter, defaultdict

import config

def get_label_from_file_name(data_dir, file_name, sep_char):
    fnametostr = ''.join(file_name).replace(data_dir, '')
    ind_start = int(fnametostr.index(sep_char)) + 1
    ind_end = int(fnametostr[ind_start:].index('_')) + ind_start
    label = int(fnametostr[ind_start:ind_end])
    label-=1
    return label

def read_utd_data(data_dir):
  data, labels, subjects = [], [], []
  for file_name in listdir(data_dir):
    try:
      file_path = join(data_dir, file_name)
      sequence = sio.loadmat(file_path)['d_skel']
      sequence = sequence.transpose(2,0,1)
      label = get_label_from_file_name(data_dir, file_name, 'a')
      subject = get_label_from_file_name(data_dir, file_name, 's')
      data.append(sequence)
      labels.append(label)
      subjects.append(subject)
    except Exception as E:
      print(E)
  print(list(set(labels)))
  print(list(set(subjects)))
  return data, labels, subjects

def get_data():

    data, labels, subjects = read_utd_data(config.UTD_DATA_DIR)

    for sequence in data:
        if np.any(np.isnan(sequence)):
            raise ValueError

    lens = [sequence.shape[0] for sequence in data]
    print(sum(lens)/len(lens))

    def drop_indices(ls, indices):
        return [ls[i] for i in range(len(ls)) if i not in indices]

    def remove_anomalies(data, labels, subjects):
        frames_to_drop = defaultdict(list)
        dropped = 0
        for sequence_ind, sequence in enumerate(data):
            for frame_ind in range(sequence.shape[0]):
                for joint in sequence[frame_ind]:
                    assert joint.shape == (3, )
                    if joint[0]>=1000 or joint[1]>=1000 or joint[2]>=1000:
                        frames_to_drop[sequence_ind].append(frame_ind)
                        dropped+=1
        for k, v in frames_to_drop.items():
            dropped-=len(v)
        assert dropped == 0
        print('Dropping frames:', frames_to_drop)
        for sequence_ind in range(len(data)):
            data[sequence_ind] = np.delete(data[sequence_ind], frames_to_drop[sequence_ind], axis=0)
        return data, labels, subjects

    def check_anomalies(data):
        for sequence in data:
            for frame in sequence:
                for joint in frame:
                    if joint[0]>=1000 or joint[1]>=1000 or joint[2]>=1000:
                        raise ValueError

    data, labels, subjects = remove_anomalies(data, labels, subjects)
    check_anomalies(data)

    return data, labels, subjects