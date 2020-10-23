import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from src.FeatureSelector import FeatureSelector
from DataGenerator import generate_data, get_one_hot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Dataset parameters
N_TRAIN_SAMPLES = 512
N_VAL_SAMPLES = 256
N_TEST_SAMPLES = 1024
N_FEATURES = 10
FEATURE_SHAPE = (10,)
dataset_label = "XOR_"

# Training parapmeters
data_batch_size = 32
mask_batch_size = 32
# final batch_size is data_batch_size x mask_batch_size
s = 5  # size of optimal subset that we are looking for
s_p = 2  # number of flipped bits in a mask when looking around m_opt
phase_2_start = 6000  # after how many batches phase 2 will begin
max_batches = 15000  # how many batches if the early stopping condition not satisfied
early_stopping_patience = 600  # how many patience batches (after phase 2 starts)
# before the training stops

# Generate data
X_tr, y_tr = generate_data(n=N_TRAIN_SAMPLES, seed=0)
X_val, y_val = generate_data(n=N_VAL_SAMPLES, seed=0)
X_te, y_te = generate_data(n=N_TEST_SAMPLES, seed=0)

# Get one hot encoding of the labels
y_tr = get_one_hot(y_tr.astype(np.int8), 4)
y_te = get_one_hot(y_te.astype(np.int8), 4)
y_val = get_one_hot(y_val.astype(np.int8), 4)

fs = FeatureSelector(FEATURE_SHAPE, s, data_batch_size, mask_batch_size, str_id=dataset_label)
fs.create_dense_operator([60, 30, 20, 4], "softmax", metrics=[keras.metrics.CategoricalAccuracy()],
                       error_func=K.categorical_crossentropy)
fs.operator.set_early_stopping_params(phase_2_start, patience_batches=early_stopping_patience, minimize=True)

fs.create_dense_selector([100, 50, 10, 1])
fs.create_mask_optimizer(epoch_condition=phase_2_start, perturbation_size=s_p)
fs.train_networks_on_data(X_tr, y_tr, max_batches, val_data=(X_val, y_val))

importances, optimal_mask = fs.get_importances(return_chosen_features=True)
optimal_subset = np.nonzero(optimal_mask)
test_performance = fs.operator.test_one(X_te, optimal_mask[None,:], y_te)
print("Importances: ", importances)
print("Optimal_subset: ", optimal_subset)
print("Test performance (CE): ", test_performance[0])
print("Test performance (ACC): ", test_performance[1])