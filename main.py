import numpy as np
from os import listdir
from collections import defaultdict
from os.path import join

import config

def get_label_from_file_name(data_dir, file_name, sep_char):
    fnametostr = ''.join(file_name).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    label-=1
    return label

def augment_labels(labels):
    curr = 0
    hash = dict()
    res = []
    for label in labels:
        if label not in hash:
            hash[label] = curr
            curr+=1
        res.append(hash[label])
    return res

def read_msr_data(data_dir, data_1, labels_1, subjects_1, data_2, labels_2, subjects_2, data_3, labels_3, subjects_3):
    file_names = np.array([d.strip() for d in sorted(listdir(data_dir))])
    file_paths = np.array([join(data_dir, d) for d in sorted(listdir(data_dir))])
    for file_name, file_path in zip(file_names, file_paths):
        if file_name in config.NOISY_SEQUENCES:
          print(f'{file_name} is a noisy sequence')
          continue
        sequence = np.loadtxt(file_path, dtype=np.float64)[:, :3]
        num_frames = (sequence.shape[0])//20
        sequence = sequence.reshape((num_frames, config.NUM_JOINTS, 3))
        label = get_label_from_file_name(data_dir=data_dir, file_name=file_path, sep_char='a')
        subject = get_label_from_file_name(data_dir=data_dir, file_name=file_path, sep_char='s')
        if label in config.action_set_1:
          data_1.append(sequence)
          labels_1.append(label)
          subjects_1.append(subject)
        if label in config.action_set_2:
          data_2.append(sequence)
          labels_2.append(label)
          subjects_2.append(subject)
        if label in config.action_set_3:
          data_3.append(sequence)
          labels_3.append(label)
          subjects_3.append(subject)
    return data_1, labels_1, subjects_1, data_2, labels_2, subjects_2, data_3, labels_3, subjects_3

def get_data():
    data_1, labels_1, subjects_1 = [], [], []
    data_2, labels_2, subjects_2 = [], [], []
    data_3, labels_3, subjects_3 = [], [], []

    data_1, labels_1, subjects_1, data_2, labels_2, subjects_2, data_3, labels_3, subjects_3 = read_msr_data(config.DATA_DIR, data_1, labels_1, subjects_1, data_2, labels_2, subjects_2, data_3, labels_3, subjects_3)

    labels_1 = augment_labels(labels=labels_1)
    labels_2 = augment_labels(labels=labels_2)
    labels_3 = augment_labels(labels=labels_3)

    if config.ACTION_SET == 1:
        data_use, labels_use, subjects_use = data_1, labels_1, subjects_1
    if config.ACTION_SET == 2:
        data_use, labels_use, subjects_use = data_2, labels_2, subjects_2
    if config.ACTION_SET == 3:
        data_use, labels_use, subjects_use = data_3, labels_3, subjects_3

    for sequence in data_use:
        if np.any(np.isnan(sequence)):
            raise ValueError
        
    lens = [sequence.shape[0] for sequence in data_use]
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

    data_use, labels_use, subjects_use = remove_anomalies(data_use, labels_use, subjects_use)
    check_anomalies(data_use)
        
    return data_use, labels_use, subjects_use
    

