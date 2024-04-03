import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from utils import EnsembleMemberModel, get_device

device = get_device()

class SVMClassifierModel(EnsembleMemberModel):

    def init_model(self, X_train, X_test, y_train, y_test, num_classes):
      self.num_classes = num_classes
      self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
      self.model = SVC()

    def fit(self):
      param_grid = {
          'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
          'degree': [2, 3, 4, 5],  # Degree of the polynomial kernel
          'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],  # Kernel coefficient
      }
      svm = SVC()
      grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
      grid_search.fit(self.X_train, self.y_train)
      self.model = SVC(
          C=grid_search.best_params_['C'],
          kernel='rbf',
          degree=grid_search.best_params_['degree'],
          gamma=grid_search.best_params_['gamma']
      )
      self.model.fit(self.X_train, self.y_train)


    def get_predictions(self):
      predictions = self.model.predict(self.X_test)
      accuracy = accuracy_score(self.y_test, predictions)
      print('SVM Accuracy:', accuracy)
      return torch.tensor(predictions, dtype=torch.long, device=device), accuracy