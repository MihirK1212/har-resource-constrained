import copy
import torch

from algorithms import statistical_moments_aggregation, window_division_aggregation, skelemotion_aggregation, dtw_aggregation
import utils
import config

device = utils.get_device()

def get_mean_feature_vector(sequence, aggregation_method):

    agg_sequence = copy.deepcopy(sequence)

    if aggregation_method == config.STATISTICAL_MOMENTS:
      return statistical_moments_aggregation.statistical_moments_aggregation(sequence=agg_sequence)

    elif aggregation_method == config.WINDOW_DIVISION:
      return window_division_aggregation.window_division_aggregation(sequence=agg_sequence)

    elif aggregation_method == config.SKELEMOTION:
      return skelemotion_aggregation.skelemotion_aggregation(sequence=agg_sequence)

    elif aggregation_method == config.DTW:
      return dtw_aggregation.dtw_aggregation(sequence=agg_sequence)

    else:
      raise ValueError
    
def get_parsed_data(data_, labels_, subjects_, train_subjects, aggregation_method):

    data, labels, subjects = copy.deepcopy(data_), copy.deepcopy(labels_), copy.deepcopy(subjects_)

    X_train, y_train = [], []
    X_test, y_test = [], []

    if config.AUGMENT_DATA:
      augmented_data, augmented_labels, augmented_subjects = [], [], []
      for sequence, label, subject in zip(data, labels, subjects):
        if subject in train_subjects:
          curr_augmented_data = utils.get_augmented_sequences(sequence)
          curr_augmented_labels = [label]*len(curr_augmented_data)
          curr_augmented_subjects = [subject]*len(curr_augmented_data)
          assert len(curr_augmented_data) == len(curr_augmented_labels) and len(curr_augmented_labels) == len(curr_augmented_subjects)
          assert curr_augmented_labels[0] == curr_augmented_labels[-1] and curr_augmented_labels[0] == label
          assert curr_augmented_subjects[0] == curr_augmented_subjects[-1] and curr_augmented_subjects[0] == subject
          augmented_data.extend(curr_augmented_data)
          augmented_labels.extend(curr_augmented_labels)
          augmented_subjects.extend(curr_augmented_subjects)
      data.extend(augmented_data)
      labels.extend(augmented_labels)
      subjects.extend(augmented_subjects)

    for sequence, label, subject in zip(data, labels, subjects):
        f = get_mean_feature_vector(sequence=sequence, aggregation_method=aggregation_method)
        if subject in train_subjects:
            X_train.append(f)
            y_train.append(label)
        else:
            X_test.append(f)
            y_test.append(label)

    X_train, y_train = utils.shuffle(X_train, y_train)
    X_test, y_test = utils.shuffle(X_test, y_test)
    X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.tensor(y_train), torch.tensor(y_test)
    X_train, X_test = torch.tensor(X_train).to(torch.float64).to(device), torch.tensor(X_test).to(torch.float64).to(device)
    y_train, y_test = torch.tensor(y_train).to(torch.long).to(device), torch.tensor(y_test).to(torch.long).to(device)
    return {
        config.X_TRAIN_KEY: X_train,
        config.X_TEST_KEY: X_test,
        config.Y_TRAIN_KEY: y_train,
        config.Y_TEST_KEY: y_test
      }