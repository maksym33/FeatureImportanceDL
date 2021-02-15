import numpy as np
from tensorflow import keras
from .DataImporter import DataImporter
from .MaskOptimizer import MaskOptimizer
from .util import get_random_masks_different_s


class DataGenerator():
    def __init__(self, data_importer, mask_optimizer):
        self.data_importer = data_importer
        self.mask_optimizer = mask_optimizer


class TaskDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_importer, mask_optimizer, mode="train"):
        'Initialization'
        self.mode = mode
        self.data_importer = data_importer
        self.mask_optimizer = mask_optimizer
        self.data_batch_size = data_importer.batch_size
        self.mask_batch_size = mask_optimizer.batch_size
        self.data_shape = data_importer.data_shape
        self.data_size = data_importer.data_size
        self.mask_shape = data_importer.mask_shape
        self.mask_size = data_importer.mask_size
        assert self.mask_size == self.data_size  # data_shape and mask_shape without respective batch_sizes

        if self.mode == "train":
            self.get_batch_fn = self.data_importer.get_train_batch
            self.length = self.data_importer.n_train_samples
        elif self.mode == "val":
            self.get_batch_fn = self.data_importer.get_val_batch
            self.length = self.data_importer.n_val_samples
        elif self.mode == "test":
            self.get_batch_fn = self.data_importer.get_test_batch
            self.length = self.data_importer.n_test_samples

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.length

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X, y = self.get_batch_fn(index)
        s_per_mask = self.mask_optimizer.get_s(self.mask_batch_size)
        m = get_random_masks_different_s(self.data_importer.data_shape, s_per_mask, dtype='uint')
        return [X, m], y

    def on_epoch_start(self):
        'Updates indexes after each epoch'
        self.data_importer.on_epoch_start()
