action_set_1 = [1, 2, 4, 5, 9, 12, 17, 19]
action_set_2 = [0, 3, 6, 7, 8, 10, 11, 13]
action_set_3 = [5, 13, 14, 15, 16, 17, 18, 19]

label_set = list(set(action_set_1 + action_set_2 + action_set_3))
assert len(label_set) == 20

MSR_ACTION_3D = 'msraction3d'
URT_MHAD = 'utd_mhad'
DATASET = URT_MHAD

MSR_DATA_DIR = './msraction3d' 
UTD_DATA_DIR = './utdmhad'
BEST_NN_MODEL_PATH = '/content/drive/My Drive/BTP/ensemble_best_nn_model.pt'

NUM_JOINTS = 20

ACTION_SET = 2

NOISY_SEQUENCES = [
    'a07_s04_e01_skeleton.txt',
    'a13_s02_e02_skeleton.txt',
    'a13_s06_e01_skeleton.txt',
    'a13_s06_e02_skeleton.txt',
    'a13_s09_e02_skeleton.txt',
    'a20_s02_e01_skeleton.txt',
    'a20_s02_e02_skeleton.txt',
    'a20_s02_e03_skeleton.txt',
    'a20_s06_e01_skeleton.txt',
    'a20_s08_e02_skeleton.txt'
]

AUGMENT_DATA = False
APPLY_SEGMENT_DISTANCE_NORMALIZATION = False
ROOT_JOINT_INDEX = 3

STATISTICAL_MOMENTS = 'statistical_moments'
WINDOW_DIVISION = 'window_division'
SKELEMOTION = 'skelemotion'
DTW = 'dtw'
USE_AGGREGATION_METHODS = [
    STATISTICAL_MOMENTS,
    WINDOW_DIVISION
]

NUM_WINDOWS = 3
WINDOW_DIVISION_COMBINE_USING_MEAN = False

NUM_OPTUNA_TRIALS = 10

ENSEMBLE_MULTIMETHOD_COMBINED_PREDICTION = True

SEED = 42
TOLERANCE = 0.1

X_TRAIN_KEY = 'X_train'
X_TEST_KEY = 'X_test'
Y_TRAIN_KEY = 'y_train'
Y_TEST_KEY = 'y_test'

