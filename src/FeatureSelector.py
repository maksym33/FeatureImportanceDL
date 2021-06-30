import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from .MaskOptimizer import MaskOptimizer
from .Operator import OperatorNetwork
from .Selector import SelectorNetwork

logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)


def mean_squared_error(y_true, y_pred):
    return K.mean((y_true - y_pred) * (y_true - y_pred), axis=1)


def tf_mean_ax_0(losses):
    return tf.reduce_mean(losses, axis=0)


def progressbar(it, prefix="", size=60):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print("\r%s[%s%s] %i/%i" % (prefix, "#" * x, "." * (size - x), j, count), end=" ")

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print()


class FeatureSelector():
    def __init__(self, data_shape, unmasked_data_size, data_batch_size, mask_batch_size, str_id="",
                 epoch_on_which_selector_trained=8):
        self.data_shape = data_shape
        self.data_size = np.zeros(data_shape).size
        self.unmasked_data_size = unmasked_data_size
        self.logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%m%d-%H%M%S"))
        self.data_batch_size = data_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = mask_batch_size * data_batch_size
        self.str_id = str_id
        self.prev_mopt_condition = False
        self.epoch_on_which_selector_trained = epoch_on_which_selector_trained

    def create_dense_operator(self, arch, activation, metrics=None, error_func=mean_squared_error, es_patience=800):
        self.operator = OperatorNetwork(self.data_batch_size, self.mask_batch_size, self.logdir + "operator" + self.str_id)
        print("Creating operator model")
        self.operator.create_dense_model(self.data_shape, arch, activation)
        print("Compiling operator")
        self.operator.compile_model(error_func, tf.reduce_mean, tf_mean_ax_0, metrics)
        print("Created operator")

    def create_conv_operator(self, filters, kernels, dense_arch, activation, img_shape=None, channels=1, padding="same",
                           metrics=None, error_func=None, es_patience=800):
        self.operator = OperatorNetwork(self.data_batch_size, self.mask_batch_size, self.logdir + "operator" + self.str_id)
        print("Creating operator model")
        if channels == 1:
            self.operator.create_1ch_conv_model(self.data_shape, image_shape=img_shape, filter_sizes=filters,
                                              kernel_sizes=kernels, dense_arch=dense_arch, padding=padding,
                                              last_activation=activation)
        else:
            self.operator.create_2ch_conv_model(self.data_shape, image_shape=img_shape, filter_sizes=filters,
                                              kernel_sizes=kernels, dense_arch=dense_arch, padding=padding,
                                              last_activation=activation)
        print("Compiling operator")
        self.operator.compile_model(error_func, tf.reduce_mean, tf_mean_ax_0, metrics)
        print("Created operator")

    def create_dense_selector(self, arch):
        self.selector = SelectorNetwork(self.mask_batch_size, tensorboard_logs_dir=self.logdir + "selector_" + self.str_id)
        self.selector.create_dense_model(self.data_shape, arch)
        self.selector.compile_model()

    def create_mask_optimizer(self, epoch_condition=5000, maximize_error=False, record_best_masks=False,
                              perturbation_size=2, use_new_optimization=False):
        self.mopt = MaskOptimizer(self.mask_batch_size, self.data_shape, self.unmasked_data_size,
                                  epoch_condition=epoch_condition, perturbation_size=perturbation_size)
        self.selector.sample_weights = self.mopt.get_mask_weights(self.epoch_on_which_selector_trained)

    def test_networks_on_data(self, x, y, masks):
        # x,y = self.udg.get_batch(number_of_data_batches)
        m = masks
        losses = self.operator.test_one(x, m, y)
        target_shape = (len(y), len(masks))
        losses = self.operator.get_per_mask_loss(target_shape)
        print("SN targets: " + str(losses))
        # print("SN mean targets: "+str(np.mean(losses,axis=0)))
        sn_preds = np.squeeze(self.selector.predict(m))
        print("SN preds: " + str(sn_preds))
        return losses

    def train_networks_on_data(self, x_tr, y_tr, number_of_batches, val_data=None, val_freq=16):
        use_val_data = True
        if val_data is None:
            use_val_data = False
        X_val = None
        y_val = None
        if (use_val_data is True):
            X_val = val_data[0]
            y_val = val_data[1]

        for i in progressbar(range(number_of_batches), "Training batch: ", 50):
            mopt_condition = self.mopt.check_condiditon()

            random_indices = np.random.randint(0, len(x_tr), self.data_batch_size)
            x = x_tr[random_indices, :]
            y = y_tr[random_indices]
            selector_train_condition = ((self.operator.epoch_counter % self.epoch_on_which_selector_trained) == 0)
            m = self.mopt.get_new_mask_batch(self.selector.model, self.selector.best_performing_mask,
                                             gen_new_opt_mask=selector_train_condition)

            self.operator.train_one(x, m, y)
            losses = self.operator.get_per_mask_loss()
            losses = losses.numpy()
            self.selector.append_data(m, losses)
            if (selector_train_condition):
                self.selector.train_one(self.operator.epoch_counter, mopt_condition)

            self.prev_mopt_condition = mopt_condition
            if (use_val_data is True and self.operator.epoch_counter % val_freq == 0):
                self.operator.validate_one(X_val, m, y_val)
            if (self.operator.useEarlyStopping is True and self.operator.ES_stop_training is True):
                print("Activate early stopping at training epoch/batch: " + str(self.operator.epoch_counter))
                print("Loading weights from epoch: " + str(self.operator.ES_best_epoch))
                self.operator.model.set_weights(self.operator.ES_best_weights)
                break

    def get_importances(self, return_chosen_features=True):
        features_opt_used = np.squeeze(
            np.argwhere(self.mopt.get_opt_mask(self.unmasked_data_size, self.selector.model, 12) == 1))
        m_best_used_features = np.zeros((1, self.data_size))
        m_best_used_features[0, features_opt_used] = 1
        grad_used_opt = -MaskOptimizer.gradient(self.selector.model, m_best_used_features)[0][0, :]
        importances = grad_used_opt
        if(return_chosen_features==False):
            return importances
        else:
            optimal_mask = m_best_used_features[0]
            return importances, optimal_mask
