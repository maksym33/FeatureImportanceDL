import logging
import os
import numpy as np
from .config import FIDL_RS,FIDL_LOG_DIR, FIDL_MODEL_DIR, FIDL_TENSORBOARD_LOG_DIR
from .util import get_one_hot
from sklearn.model_selection import StratifiedKFold, KFold


class DataImporter:
    def __init__(self, dataset_name, batch_size=32, path=None):
        self.dataset_name = dataset_name
        logging.debug("Loading dataset name: " + str(self.dataset_name))
        if dataset_name == 'XOR':
            self.init_XOR(path)
        elif self.dataset_name == "mnist" or self.dataset_name == "MNIST":
            self.init_MNIST(path)
        else:
            raise ValueError("Dataset name ", dataset_name, " not implemented.")
        self.batch_size = batch_size
        self.X, self.y, self.X_test, self.y_test = None, None, None, None
        self.X_train, self.y_train, self.X_val, self.y_val = None, None, None, None
        self.y_labels, self.y_test_labels = None, None
        self.n_classes = None
        self.fold = None
        self.n_train_samples, self.n_val_samples, self.n_test_samples = None, None, None
        self.n_epoch, self.n_batch = None, None
        self.train_idx, self.n_train_batches = None, None
        self.val_idx, self.test_idx = None, None

    def get_batch(self,mode, batch_number):
        if mode=='train':
            return self.get_train_batch(batch_number)
        elif mode=='val':
            return self.get_val_batch(batch_number)
        elif mode=='test':
            return self.get_test_batch(batch_number)
        else:
            return None

    def get_test_batch(self, batch_number):
        idx = self.test_idx[self.batch_size * batch_number:self.batch_size * (batch_number + 1)]
        return self.X_test[idx], self.y_test[idx]

    def get_val_batch(self,batch_number):
        idx = self.val_idx[self.batch_size * batch_number:self.batch_size * (batch_number + 1)]
        return self.X_val[idx], self.y_val[idx]

    def get_train_batch(self, batch_number=None):
        if batch_number is None:
            if self.n_batch >= self.n_train_batches:
                self.on_epoch_start()
            idx = self.train_idx[self.batch_size * self.n_batch:self.batch_size * (self.n_batch + 1)]
            self.n_batch += 1
        else:
            assert batch_number < self.n_train_batches
            idx = self.train_idx[self.batch_size * batch_number:self.batch_size * (batch_number + 1)]
        return self.X_train[idx], self.y_train[idx]

    def on_epoch_start(self):
        'Updates indexes after each epoch'
        self.train_idx = np.arange(self.n_train_samples)
        np.random.shuffle(self.train_idx)
        if self.n_epoch is None:
            self.n_epoch = 0
        else:
            self.n_epoch += 1
        #logging.debug("Data importer starts epoch: " + str(self.n_epoch) + ".")
        self.n_batch = 0
        return

    def load_data(self, **kwargs):
        logging.info("Dataset args: ", kwargs)
        if self.dataset_name == "XOR":
            self.load_XOR_data(kwargs=kwargs)
        elif self.dataset_name == "mnist" or self.dataset_name == "MNIST":
            self.load_MNIST_data(kwargs=kwargs)
        else:
            raise ValueError("Dataset name ", self.dataset_name, " not implemented.")
        os.makedirs(os.path.join(FIDL_MODEL_DIR, self.dataset_name),exist_ok=True)
        os.makedirs(os.path.join(FIDL_TENSORBOARD_LOG_DIR, self.dataset_name),exist_ok=True)

        self.data_size = np.prod(self.data_shape)

    @staticmethod
    def convert_labels_to_consecutive(labels, n_classes, verbose=True):
        unq_labels = np.unique(labels)
        assert len(unq_labels == n_classes)
        logging.debug("During converting labels found unique: " + str(unq_labels))
        label_idxs = [np.argwhere(labels == lab) for lab in unq_labels]
        for i in range(n_classes):
            if verbose:
                logging.info("Converting label " + str(unq_labels[i]) + " to " + str(i) + ".")
            else:
                logging.debug("Converting label " + str(unq_labels[i]) + " to " + str(i) + ".")
            labels[label_idxs[i]] = i
        return labels

    def load_one_hot_label_encoding(self):
        logging.debug("Changing y to one-hot encoding.")
        self.y_labels = self.y
        self.y_test_labels = self.y_test
        self.y_labels = DataImporter.convert_labels_to_consecutive(self.y_labels, self.n_classes)
        self.y_test_labels = DataImporter.convert_labels_to_consecutive(self.y_test_labels, self.n_classes, False)
        self.y = get_one_hot(self.y_labels, self.n_classes)
        self.y_test = get_one_hot(self.y_test_labels, self.n_classes)

    def divideKFold(self, K):

        logging.debug("Setting kfold for k=" + str(K))
        if self.n_classes is None:
            # regression or only 2 classes
            logging.debug("Using NOT-stratified kfold.")
            self.kfold = KFold(n_splits=K, random_state=0, shuffle=True)
        else:
            logging.debug("Using stratified kfold.")
            self.kfold = StratifiedKFold(n_splits=K, random_state=0, shuffle=True)
        self.max_fold = K - 1
        self.folds_list = [idx for idx in self.kfold.split(self.X, self.y if self.y_labels is None else self.y_labels)]

    def newFold(self, n=None):
        if n is None:
            if self.fold is None:
                self.fold = 0
            else:
                self.fold += 1
                self.fold = self.fold % self.max_fold
        else:
            assert n < self.max_fold and n >= 0
            self.fold = n
        logging.debug("Current fold set to " + str(self.fold))
        self.fold_train_idx, self.fold_val_idx = self.folds_list[self.fold]
        self.X_train = self.X[self.fold_train_idx]
        self.X_val = self.X[self.fold_val_idx]
        self.y_train = self.y[self.fold_train_idx]
        self.y_val = self.y[self.fold_val_idx]
        if not (self.y_labels is None):
            self.y_train_labels = self.y_labels[self.fold_train_idx]
            self.y_val_labels = self.y_labels[self.fold_val_idx]
        self.n_train_samples = self.X_train.shape[0]
        self.n_val_samples = self.X_val.shape[0]
        self.n_test_samples = self.X_test.shape[0]
        self.val_idx = np.arange(self.n_val_samples)
        self.test_idx = np.arange(self.n_test_samples)
        logging.debug("Counted " + str(self.n_train_samples) + "/"+ str(self.n_val_samples)
                      + "/" + str(self.n_test_samples) + " train/val/test samples.")
        self.n_train_batches = np.ceil(1.0 * self.n_train_samples / self.batch_size).astype('int')
        logging.debug("Counted " + str(self.n_train_batches) + " training batches.")

    def print_summary(self):
        if not(self.n_classes is None):
            logging.info("Number of classes: " + str(self.n_classes))
        logging.info("X shape: " + str(self.X.shape))
        logging.info("y shape: " + str(self.y.shape))
        logging.info("X_test shape: " + str(self.X_test.shape))
        logging.info("y_test shape: " + str(self.y_test.shape))
        if not (self.X_train is None):
            logging.info("X_train shape: " + str(self.X_train.shape))
            logging.info("y_train shape: " + str(self.y_train.shape))
            logging.info("X_val shape: " + str(self.X_val.shape))
            logging.info("y_val shape: " + str(self.y_val.shape))

    def init_MNIST(self, path):
        return

    def load_MNIST_data(self, kwargs):
        from tensorflow.keras.datasets.mnist import load_data as load_mnist
        logging.info("Loading MNIST dataset.")
        assert 'use2D' in kwargs
        if 'digits' in kwargs:
            used_digits = kwargs['digits']
        else:
            used_digits = [i for i in range(10)]
        logging.info('Used digits: ' + str(used_digits))
        (self.X, self.y), (self.X_test, self.y_test) = load_mnist()
        used_train_indices = np.concatenate([np.squeeze(np.argwhere(self.y == d)) for d in used_digits], axis=0)
        used_test_indices = np.concatenate([np.squeeze(np.argwhere(self.y_test == d)) for d in used_digits], axis=0)
        self.X = self.X[used_train_indices]
        self.y = self.y[used_train_indices]
        self.X_test = self.X_test[used_test_indices]
        self.y_test = self.y_test[used_test_indices]
        self.data_shape = (28,28)
        if kwargs['use2D'] is False:
            self.data_shape=(784,)
            self.X = self.X.reshape((self.X.shape[0], -1))
            self.X_test = self.X_test.reshape((self.X_test.shape[0], -1))
        self.X = self.X.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        self.n_classes = len(used_digits)
        self.used_digits = used_digits
        self.output_shape = (self.n_classes,)
        self.load_one_hot_label_encoding()

    def init_XOR(self, path):
        return

    def load_XOR_data(self, kwargs):
        """
        Generate data (X,y)
        Args:
        Return:
            X(float): [n,10].
            y(float): n dimensional array.
        Taken from https://github.com/Jianbo-Lab/CCM
        See:
        http://papers.nips.cc/paper/7270-kernel-feature-selection-via-conditional-covariance-minimization.pdf
        for details.
        """
        n_train = kwargs['n_train_samples']
        n_test = kwargs['n_test_samples']
        n = n_train + n_test
        X = FIDL_RS.randn(n, 10)
        y = np.zeros(n,dtype='int')
        splits = np.linspace(0, n, num=8 + 1, dtype=int)
        signals = [[1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [1, -1, 1]]
        for i in range(8):
            X[splits[i]:splits[i + 1], :3] += np.array([signals[i]])
            y[splits[i]:splits[i + 1]] = i // 2
        perm_inds = FIDL_RS.permutation(n)
        self.data_shape=(10,)
        self.n_classes = 4
        self.output_shape=(4,)
        self.X, self.y = X[perm_inds[:n_train]], y[perm_inds[:n_train]]
        self.X_test, self.y_test = X[perm_inds[n_train:]], y[perm_inds[n_train:]]
        self.load_one_hot_label_encoding()