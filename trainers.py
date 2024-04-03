from models.knn import KNNClassifierModel
from models.neural_network import NeuralNetworkClassifierModel
from models.random_forest import RandomForestClassifierModel
from models.svm import SVMClassifierModel
from models.xgb import XGBClassifierModel

import config

statistical_moments_trainers = [
          {
              'model': XGBClassifierModel(),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'xgb_statistical_moments'
          },
          {
              'model': NeuralNetworkClassifierModel(lr_decay_epochs=100),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'nn1_statistical_moments'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.01, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'nn2_statistical_moments'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs = 200, num_epochs = 200, num_hidden_layers = 2, hidden_layer_dim_factor=[1, 1]),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'nn3_statistical_moments'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'nn4_statistical_moments'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_end_factor=1.0, lr_decay_epochs=100, num_epochs=100),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'nn5_statistical_moments'
          },
          {
              'model': RandomForestClassifierModel(),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'rf_statistical_moments'
          },
          {
              'model': KNNClassifierModel(),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'knn_statistical_moments'
          },
          {
              'model': SVMClassifierModel(),
              'aggregation_method': config.STATISTICAL_MOMENTS,
              'name': 'svm_statistical_moments'
          }
]


window_division_trainers = [
         {
              'model': XGBClassifierModel(),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'xgb_window_division'
          },
          {
              'model': NeuralNetworkClassifierModel(lr_decay_epochs=100),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'nn1_window_division'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.01, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'nn2_window_division'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs = 200, num_epochs = 200, num_hidden_layers = 2, hidden_layer_dim_factor=[1, 1]),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'nn3_window_division'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'nn4_window_division'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_end_factor=1.0, lr_decay_epochs=100, num_epochs=100),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'nn5_window_division'
          },
          {
              'model': RandomForestClassifierModel(),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'rf_window_division'
          },
          {
              'model': SVMClassifierModel(),
              'aggregation_method': config.WINDOW_DIVISION,
              'name': 'svm_window_division'
          },
]

skelemotion_trainers = [
    {
              'model': XGBClassifierModel(),
              'aggregation_method': config.SKELEMOTION,
              'name': 'xgb_skelemotion'
          },
          {
              'model': NeuralNetworkClassifierModel(lr_decay_epochs=100),
              'aggregation_method': config.SKELEMOTION,
              'name': 'nn1_skelemotion'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.01, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.SKELEMOTION,
              'name': 'nn2_skelemotion'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs = 200, num_epochs = 200, num_hidden_layers = 2, hidden_layer_dim_factor=[1, 1]),
              'aggregation_method': config.SKELEMOTION,
              'name': 'nn3_skelemotion'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.SKELEMOTION,
              'name': 'nn4_skelemotion'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_end_factor=1.0, lr_decay_epochs=100, num_epochs=100),
              'aggregation_method': config.SKELEMOTION,
              'name': 'nn5_skelemotion'
          },
          {
              'model': RandomForestClassifierModel(),
              'aggregation_method': config.SKELEMOTION,
              'name': 'rf_skelemotion'
          },
          {
              'model': KNNClassifierModel(),
              'aggregation_method': config.SKELEMOTION,
              'name': 'knn_skelemotion'
          },
          {
              'model': SVMClassifierModel(),
              'aggregation_method': config.SKELEMOTION,
              'name': 'svm_skelemotion'
          }
]

dtw_trainers = [
          {
              'model': XGBClassifierModel(),
              'aggregation_method': config.DTW,
              'name': 'xgb_dtw'
          },
          {
              'model': NeuralNetworkClassifierModel(lr_decay_epochs=100),
              'aggregation_method': config.DTW,
              'name': 'nn1_dtw'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.01, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.DTW,
              'name': 'nn2_dtw'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs = 200, num_epochs = 200, num_hidden_layers = 2, hidden_layer_dim_factor=[1, 1]),
              'aggregation_method': config.DTW,
              'name': 'nn3_dtw'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_epochs=100, num_epochs=200),
              'aggregation_method': config.DTW,
              'name': 'nn4_dtw'
          },
          {
              'model': NeuralNetworkClassifierModel(lr=0.1, lr_decay_end_factor=1.0, lr_decay_epochs=100, num_epochs=100),
              'aggregation_method': config.DTW,
              'name': 'nn5_dtw'
          },
          {
              'model': RandomForestClassifierModel(),
              'aggregation_method': config.DTW,
              'name': 'rf_dtw'
          },
          {
              'model': KNNClassifierModel(),
              'aggregation_method': config.DTW,
              'name': 'knn_dtw'
          },
          {
              'model': SVMClassifierModel(),
              'aggregation_method': config.DTW,
              'name': 'svm_dtw'
          }
]
