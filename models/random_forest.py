import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils import EnsembleMemberModel, get_device

device = get_device()

class RandomForestClassifierModel(EnsembleMemberModel):

    def init_model(self, X_train, X_test, y_train, y_test, num_classes):
      self.num_classes = num_classes
      self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
      self.model = RandomForestClassifier()

    def fit(self):
      self.model.fit(self.X_train, self.y_train)

    def get_predictions(self):
      predictions = self.model.predict(self.X_test)
      accuracy = accuracy_score(self.y_test, predictions)
      print('RF Accuracy:', accuracy)
      return torch.tensor(predictions, dtype=torch.long, device=device), accuracy