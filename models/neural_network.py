import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as torch_data_utils
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score
from torch.nn.functional import relu

from utils import CustomDataset, EnsembleMemberModel, get_device
import config

device = get_device()

class FeatureExtractorNeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_layer_dim_factor, num_classes):
      super().__init__()
      assert num_hidden_layers == len(hidden_layer_dim_factor)

      self.norm = nn.BatchNorm1d(input_dim)

      self.hidden_layers = []
      curr_dim = input_dim
      for i in range(num_hidden_layers):
        self.hidden_layers.append(nn.Linear(int(curr_dim), int(curr_dim*hidden_layer_dim_factor[i])))
        curr_dim = curr_dim*hidden_layer_dim_factor[i]

      self.output_layer = nn.Linear(int(curr_dim), num_classes)
      self.init_weights()

    def init_weights(self) -> None:
      initrange = 0.1
      for layer in self.hidden_layers:
        layer.bias.data.zero_()
        layer.weight.data.uniform_(-initrange, initrange)
      self.output_layer.bias.data.zero_()
      self.output_layer.weight.data.uniform_(-initrange, initrange)
      for p in self.parameters():
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.norm(x)
        for layer in self.hidden_layers:
          x = relu(layer(x))
        x = self.output_layer(x)
        return x

class NeuralNetworkClassifierModel(EnsembleMemberModel):

    def __init__(self, lr = 0.1, lr_decay_end_factor = 0.1, lr_decay_epochs = 50, num_epochs = 100, num_hidden_layers = 1, hidden_layer_dim_factor=[1]):
      self.lr = lr
      self.lr_decay_end_factor = lr_decay_end_factor
      self.lr_decay_epochs = lr_decay_epochs
      self.num_epochs = num_epochs
      self.num_hidden_layers = num_hidden_layers
      self.hidden_layer_dim_factor = hidden_layer_dim_factor

    def init_model(self, X_train, X_test, y_train, y_test, num_classes):
      self.num_classes = num_classes
      self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
      train_dataset = CustomDataset(X_train, y_train)
      test_dataset = CustomDataset(X_test, y_test)
      self.train_dataloader = torch_data_utils.DataLoader(train_dataset, batch_size=8, shuffle=True)
      self.test_dataloader  = torch_data_utils.DataLoader(test_dataset, batch_size=8, shuffle=False)

      self.input_dim = X_train.shape[-1]

      self.model = FeatureExtractorNeuralNetwork(
          input_dim=self.input_dim, num_hidden_layers=self.num_hidden_layers, hidden_layer_dim_factor=self.hidden_layer_dim_factor,
          num_classes=self.num_classes
      ).to(device)

      for p in self.model.parameters():
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)

      self.opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)
      self.loss_fn = torch.nn.CrossEntropyLoss()
      self.scheduler = lr_scheduler.LinearLR(self.opt, start_factor=1.0, end_factor=self.lr_decay_end_factor, total_iters=self.lr_decay_epochs)
      self.num_epochs = self.num_epochs

    def get_validation_accuracy(self):
        self.model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
              X, y = torch.tensor(X).to(torch.float32).to(device), torch.tensor(y).to(torch.long).to(device)
              output = self.model(X)
              loss = self.loss_fn(output, y)
              probabilities = nn.functional.softmax(output, dim=1)
              _, batch_predictions = torch.max(probabilities, 1)
              accuracy = (batch_predictions == y).to(torch.float64).mean().item()
              total_accuracy+=accuracy
        return total_accuracy / len(self.test_dataloader)

    def fit(self):
      max_accuracy = 0.0
      for epoch in range(self.num_epochs):
        epoch_accuracy = 0
        self.model.train()
        for X, y in self.train_dataloader:
          X, y = torch.tensor(X).to(torch.float32).to(device), torch.tensor(y).to(torch.long).to(device)
          if X.shape[0] <= 1:
              continue
          output = self.model(X)
          loss = self.loss_fn(output, y)
          self.opt.zero_grad()
          loss.backward()
          self.opt.step()

        self.scheduler.step()
        epoch_accuracy=self.get_validation_accuracy()
        if epoch_accuracy > max_accuracy:
          torch.save(self.model.state_dict(), config.BEST_NN_MODEL_PATH)
          max_accuracy = epoch_accuracy
      self.max_epoch_accuracy = max_accuracy
      self.model.load_state_dict(torch.load(config.BEST_NN_MODEL_PATH))

    def get_predictions(self):
      self.model.eval()
      predictions = []
      with torch.no_grad():
          for X, y in self.test_dataloader:
            X, y = torch.tensor(X).to(torch.float32).to(device), torch.tensor(y).to(torch.long).to(device)
            output = self.model(X)
            loss = self.loss_fn(output, y)
            probabilities = nn.functional.softmax(output, dim=1)
            _, batch_predictions = torch.max(probabilities, 1)
            batch_predictions = batch_predictions.tolist()
            predictions.extend(batch_predictions)
      predictions = torch.tensor(predictions, dtype=torch.long, device=device)
      accuracy = accuracy_score(self.y_test, predictions)
      try:
        assert abs(accuracy-self.max_epoch_accuracy) < config.TOLERANCE
      except:
        print('Warning! NN Accuracy did not match. Max Accuracy:', self.max_epoch_accuracy, ' Predictions Accuracy:', accuracy)
      print('NN Accuracy:', accuracy)
      return predictions, accuracy