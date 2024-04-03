import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from utils import EnsembleMemberModel, get_device

device = get_device()


class KNNClassifierModel(EnsembleMemberModel):

    def init_model(self, X_train, X_test, y_train, y_test, num_classes):
      self.num_classes = num_classes
      self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
      self.model = KNeighborsClassifier()

    def fit(self):
      param_grid = {
          'n_neighbors': [1, 3, 5, 7, 9],
          'weights': ['uniform', 'distance'],
          'p': [1, 2],
          'leaf_size': [10, 20, 30, 40, 50],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
          'metric': ['euclidean', 'manhattan', 'minkowski'],
          'metric_params': [{'p': 1}, {'p': 2}]
      }
      knn = KNeighborsClassifier()
      grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
      grid_search.fit(self.X_train, self.y_train)
      self.model = KNeighborsClassifier(
          n_neighbors=grid_search.best_params_['n_neighbors'],
          weights=grid_search.best_params_['weights'],
          p=grid_search.best_params_['p'],
          leaf_size=grid_search.best_params_['leaf_size'],
          algorithm=grid_search.best_params_['algorithm'],
          metric=grid_search.best_params_['metric'],
          metric_params=grid_search.best_params_['metric_params']
      )
      self.model.fit(self.X_train, self.y_train)


    def get_predictions(self):
      predictions = self.model.predict(self.X_test)
      accuracy = accuracy_score(self.y_test, predictions)
      print('KNN Accuracy:', accuracy)
      return torch.tensor(predictions, dtype=torch.long, device=device), accuracy