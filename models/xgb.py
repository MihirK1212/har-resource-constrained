import torch
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils import EnsembleMemberModel

import config

class XGBClassifierModel(EnsembleMemberModel):

    def init_model(self, X_train, X_test, y_train, y_test, num_classes):
      self.num_classes = num_classes
      self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
      self.train_d = xgb.DMatrix(X_train, label=y_train)
      self.test_d  = xgb.DMatrix(X_test, label=y_test)
      self.num_optuna_trials = config.NUM_OPTUNA_TRIALS

    def objective(self, trial):
        params = {
          'max_depth': trial.suggest_int('max_depth', 5, 25),
          'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
          'subsample': trial.suggest_float('subsample', 0.50, 1),
          'colsample_bytree': trial.suggest_float('colsample_bytree', 0.50, 1),
          'gamma': trial.suggest_int('gamma', 0, 10),
          'min_child_weight': trial.suggest_float('min_child_weight', 0, 3),
          'objective': 'multi:softmax',
          'num_class': self.num_classes,
          'random_state': config.SEED
        }
        bst = xgb.train(params, self.train_d)
        predictions = np.rint(bst.predict(self.test_d))
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy

    def fit(self):
      study = optuna.create_study(direction="maximize")
      study.optimize(self.objective, n_trials=self.num_optuna_trials, timeout=600)
      self.best_trial_accuracy = study.best_trial.value
      best_params = study.best_trial.params
      self.model = xgb.train({
          'max_depth': best_params['max_depth'],
          'learning_rate': best_params['learning_rate'],
          'subsample': best_params['subsample'],
          'colsample_bytree': best_params['colsample_bytree'],
          'gamma': best_params['gamma'],
          'min_child_weight': best_params['min_child_weight'],
          'objective': 'multi:softmax',
          'num_class': self.num_classes,
          'random_state': config.SEED
          }, self.train_d)

    def get_predictions(self):
      predictions = np.rint(self.model.predict(self.test_d))
      accuracy = accuracy_score(self.y_test, predictions)
      assert accuracy == self.best_trial_accuracy
      print('XGB Accuracy:', accuracy)
      return torch.tensor(predictions, dtype=torch.long), accuracy