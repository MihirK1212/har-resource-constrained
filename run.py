import warnings
import random
import torch
from collections import Counter
import copy

import main
import config
import data_utils
from trainers import statistical_moments_trainers, window_division_trainers, dtw_trainers, skelemotion_trainers
import utils
import config

device = utils.get_device()

class EnsembleTrial():

  def __init__(self, data, labels, subjects):
    self.data = data
    self.labels = labels
    self.subjects = subjects
    self.num_classes = len(list(set(self.labels)))

    def get_train_subjects(subjects):
      random.shuffle(subjects)
      n = len(subjects)
      half_n = n // 2
      random_half = subjects[:half_n]
      return random_half
    self.train_subjects = get_train_subjects(list(set(self.subjects)))
    print('Train Subjects:', self.train_subjects)
    print()

    self.statistical_moments_data = data_utils.get_parsed_data(self.data, self.labels, self.subjects, self.train_subjects, aggregation_method=config.STATISTICAL_MOMENTS)
    self.window_division_data = data_utils.get_parsed_data(self.data, self.labels, self.subjects, self.train_subjects, aggregation_method=config.WINDOW_DIVISION)
    self.skelemotion_data = data_utils.get_parsed_data(self.data, self.labels, self.subjects, self.train_subjects, aggregation_method=config.SKELEMOTION)
    self.dtw_data = data_utils.get_parsed_data(self.data, self.labels, self.subjects, self.train_subjects, aggregation_method=config.DTW)

    assert torch.equal(self.statistical_moments_data[config.Y_TRAIN_KEY], self.window_division_data[config.Y_TRAIN_KEY])
    assert torch.equal(self.statistical_moments_data[config.Y_TEST_KEY], self.window_division_data[config.Y_TEST_KEY])
    assert torch.equal(self.statistical_moments_data[config.Y_TRAIN_KEY], self.skelemotion_data[config.Y_TRAIN_KEY])
    assert torch.equal(self.statistical_moments_data[config.Y_TEST_KEY], self.skelemotion_data[config.Y_TEST_KEY])
    assert torch.equal(self.statistical_moments_data[config.Y_TRAIN_KEY], self.dtw_data[config.Y_TRAIN_KEY])
    assert torch.equal(self.statistical_moments_data[config.Y_TEST_KEY], self.dtw_data[config.Y_TEST_KEY])

    self.y_test = self.statistical_moments_data[config.Y_TEST_KEY]

    self.log_data_info(self.statistical_moments_data, 'Statistical Moments Data')
    self.log_data_info(self.window_division_data, 'Window Division Data')
    self.log_data_info(self.skelemotion_data, 'Skelemotion Data')
    self.log_data_info(self.dtw_data, 'DTW Data')

    self.ensemble_trainers = self.ensemble_trainers = [
            *statistical_moments_trainers,
            *window_division_trainers,
            # *skelemotion_trainers
            # *dtw_trainers
      ]

    for trainer in self.ensemble_trainers:
      model, data = trainer['model'], self.get_model_data(trainer['aggregation_method'])
      model.init_model(data[config.X_TRAIN_KEY], data[config.X_TEST_KEY], data[config.Y_TRAIN_KEY], data[config.Y_TEST_KEY], self.num_classes)

  def get_model_data(self, aggregation_method):
      if aggregation_method == config.STATISTICAL_MOMENTS:
        return self.statistical_moments_data
      elif aggregation_method == config.WINDOW_DIVISION:
        return self.window_division_data
      elif aggregation_method == config.SKELEMOTION:
        return self.skelemotion_data
      elif aggregation_method == config.DTW:
        return self.dtw_data
      else:
        raise ValueError

  def log_data_info(self, data, data_name):
    X_train, X_test, y_train, y_test = data[config.X_TRAIN_KEY], data[config.X_TEST_KEY], data[config.Y_TRAIN_KEY], data[config.Y_TEST_KEY]
    print(data_name)
    print('Number of NaN values:', torch.isnan(X_train).sum().item(), torch.isnan(X_test).sum().item())
    print('Data Shape:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print()

  def get_majority_vote(self, votes):
    return max(Counter(votes).items(), key=lambda x: x[1])[0] if Counter(votes).most_common(1)[0][1] > len(votes)//2 else votes[self.max_accuracy_trainer_index]

  def get_polled_predictions(self, model_predictions_list):
    predictions = []
    num_samples = model_predictions_list[0].shape[0]
    for s in range(num_samples):
      votes = []
      for model_predictions in model_predictions_list:
        assert len(model_predictions) == num_samples
        votes.append(int(model_predictions[s]))
      prediction = self.get_majority_vote(votes)
      predictions.append(prediction)
    return torch.tensor(predictions, dtype=torch.long, device = device)

  def get_predictions_accuracy(self, aggregation_method = None):

    print('~')
    print('For Aggregation Method:', aggregation_method)
    print()

    trainers = [trainer for trainer in self.ensemble_trainers if ((aggregation_method is None) or trainer['aggregation_method']==aggregation_method)]
    model_predictions_list = []
    max_accuracy = 0
    for trainer_index, trainer in enumerate(trainers):
      model_predictions, model_accuracy = trainer['model'].get_predictions()
      model_predictions_list.append(model_predictions)
      if model_accuracy > max_accuracy:
        max_accuracy = model_accuracy
        self.max_accuracy_trainer_index = trainer_index

    print()
    print('Best Model Accuracy:', max_accuracy)
    print('Best Accuracy Model Index:', self.max_accuracy_trainer_index)
    print()

    predictions = self.get_polled_predictions(model_predictions_list)
    predictions = predictions.to(device)
    accuracy = (predictions == self.y_test).to(torch.float64).mean().item()

    print('Ensemble Accuracy:', accuracy)
    print('~')
    print()

    if accuracy < max_accuracy:
      predictions = model_predictions_list[self.max_accuracy_trainer_index]
      predictions = predictions.to(device)
      accuracy = (predictions == self.y_test).to(torch.float64).mean().item()

    return accuracy

  def run_trial(self):
    for trainer in self.ensemble_trainers:
      if trainer['aggregation_method'] in config.USE_AGGREGATION_METHODS:
        print(f"Training {trainer['name']}")
        trainer['model'].fit()
    print()
    if config.ENSEMBLE_MULTIMETHOD_COMBINED_PREDICTION:
      return self.get_predictions_accuracy()
    else:
      max_accuracy = 0
      for aggregation_method in config.USE_AGGREGATION_METHODS:
        max_accuracy = max(max_accuracy, self.get_predictions_accuracy(aggregation_method=aggregation_method))
      return max_accuracy

data_use, labels_use, subjects_use = main.get_data()

num_trials = 10
avg_accuracy = 0
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  for i in range(num_trials):
    print('##########################################')
    et = EnsembleTrial(copy.deepcopy(data_use), copy.deepcopy(labels_use), copy.deepcopy(subjects_use))
    accuracy = et.run_trial()
    print('Trial Accuracy:', accuracy)
    print('##########################################')
    print()
    avg_accuracy+=accuracy
print()
print('Average Accuracy:', avg_accuracy/num_trials)