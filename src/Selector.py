import numpy as np
import tensorflow as tf
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

class SelectorNetwork:
    def __init__(self, mask_batch_size, tensorboard_logs_dir=""):
        self.batch_size = mask_batch_size
        self.mask_batch_size = mask_batch_size
        self.tr_loss_history = []
        self.te_loss_history = []
        self.y_pred_std_history = []
        self.y_true_std_history = []
        self.tf_logs = tensorboard_logs_dir
        self.epoch_counter = 0
        self.data_masks = None
        self.data_targets = None
        self.best_performing_mask = None
        self.sample_weights = None

    def set_label_input_params(self, y_shape, y_input_layer):
        self.label_input_layer = y_input_layer
        self.label_shape = y_shape

    def create_dense_model(self, input_shape, dense_arch):
        input_mask_layer = Input(shape=input_shape)
        x = Flatten()(input_mask_layer)
        for i in range(len(dense_arch[:-1])):
            x = Dense(dense_arch[i], activation="sigmoid")(x)
        x = Dense(dense_arch[-1], activation="linear")(x)
        self.model = Model(inputs=[input_mask_layer], outputs=x)
        print("Subject Network model built:")
        #self.model.summary()

    def named_logs(self, model, logs):
        result = {}
        try:
            iterator = iter(logs)
        except TypeError:
            logs = [logs]
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def compile_model(self):
        self.model.compile(loss='mae', optimizer='adam',
                           metrics=[self.get_y_std_metric(True), self.get_y_std_metric(False)])
        if self.tf_logs != "":
            log_path = './logs'
            self.tb_clbk = TensorBoard(self.tf_logs)
            self.tb_clbk.set_model(self.model)


    def train_one(self, epoch_number, apply_weights):  # train on data in object memory
        if apply_weights == False:
            curr_loss = self.model.train_on_batch(x=self.data_masks, y=self.data_targets)
        else:
            curr_loss = self.model.train_on_batch(x=self.data_masks, y=self.data_targets,
                                                  sample_weight=self.sample_weights)
        self.best_performing_mask = self.data_masks[np.argmin(self.data_targets, axis=0)]
        self.tr_loss_history.append(curr_loss)
        self.epoch_counter = epoch_number
        if self.tf_logs != "":
            self.tb_clbk.on_epoch_end(self.epoch_counter, self.named_logs(self.model, curr_loss))
        self.data_masks = None
        self.data_targets = None


    def append_data(self, x, y):
        if self.data_masks is None:
            self.data_masks = x
            self.data_targets = y
        else:
            self.data_masks = np.concatenate([self.data_masks, x], axis=0)
            self.data_targets = tf.concat([self.data_targets, y], axis=0)

    def test_one(self, x, y):
        y_pred = self.model.predict(x=x)
        curr_loss = self.model.test_on_batch(x=x, y=y)
        self.te_loss_history.append(curr_loss)
        # print("SN test loss: "+str(curr_loss))
        # print("SN prediction: "+str(np.squeeze(curr_loss)))
        # print("SN targets: "+str(np.squeeze(y_pred)))
        return curr_loss

    def predict(self, x):
        y_pred = self.model.predict(x=x)
        return y_pred

    def get_y_std_metric(self, ifpred=True):
        def y_pred_std_metric(y_true, y_pred):
            y_pred_std = K.std(y_pred)
            self.y_pred_std_history.append(y_pred_std)
            return y_pred_std

        def y_true_std_metric(y_true, y_pred):
            y_true_std = K.std(y_true)
            self.y_true_std_history.append(y_true_std)
            return y_true_std

        if (ifpred == True):
            return y_pred_std_metric
        else:
            return y_true_std_metric

