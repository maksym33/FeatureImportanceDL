import logging
import datetime
import os
import gc
import tracemalloc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from .DataImporter import DataImporter
from .util import create_masked_batch, get_random_masks_different_s, \
    increment_mask, reshape_masked_variable, DataSavingCallback, CleaningCallback
from .MaskOptimizer import MaskOptimizer
from .config import FIDL_TENSORBOARD_LOG_DIR, FIDL_MODEL_DIR


class ModelTrainer:

    def __init__(self, dataImporter, mask_batch_size, flip_size=None, s_guess=None, eager_mode=False):

        assert mask_batch_size % 2 == 0
        self.flip_size = flip_size
        self.eager_mode = eager_mode
        self.datetime_start = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.models = []
        self.dataImporter = dataImporter
        self.dataset_name = dataImporter.dataset_name
        self.data_batch_size = self.dataImporter.batch_size
        self.mask_batch_size = mask_batch_size
        self.batch_size = self.data_batch_size * self.mask_batch_size
        self.data_shape = self.dataImporter.X.shape[1:]
        self.n_features = np.prod(self.data_shape)
        self.task_mode = -1
        self.mode_dict = {"train": 0, "val": 1, "test": 2}
        self.new_pfs_data_present = False
        self.PFS_y_true = []
        self.PFS_X = []
        self.PFS_model_current_epoch = 0
        self.task_model_current_epoch = 0
        self.task_hist, self.pfs_hist, self.ifs_hist = [], [], []
        logging.debug("Trainer found " + str(self.n_features) + " features.")
        self.task_output_vals = tf.Variable(0.0, shape=tf.TensorShape(None))
        self.save_pfs_data = False
        #self.last_used_m = None

        self.mask_optimizer = MaskOptimizer(mask_batch_size, self.data_shape, flip_size,
                                            frac_of_rand_masks=0.5, s_guess=s_guess)
        self.train_length = np.ceil(1.0 * self.dataImporter.n_train_samples / self.data_batch_size).astype('int')
        self.val_length = np.ceil(1.0 * self.dataImporter.n_val_samples / self.data_batch_size).astype('int')
        self.test_length = np.ceil(1.0 * self.dataImporter.n_test_samples / self.data_batch_size).astype('int')
        self.length_dict = {"train": self.train_length, "val": self.val_length, "test": self.test_length}
        logging.debug("N batches per mode: " + str(self.length_dict))

        return

    def fit_phase_1(self, max_epoch, early_stop, lr_patience, val_freq=1):
        # - 1st PHASE
        # - FIT ORIGINAL TASK NN UNTIL VALIDATION LOSS STOPS INCREASING
        def data_gen(mode):
            logging.debug("Init datagen for phase 1 with mode: " + mode)
            half_mask_shape = (int(self.mask_batch_size / 2),) + self.data_shape
            logging.debug("Half Mask shape: " + str(half_mask_shape))
            while True:
                self.dataImporter.on_epoch_start()
                for i in range(self.length_dict[mode]):
                    x, y = self.dataImporter.get_batch(mode, i)
                    s_per_mask = self.mask_optimizer.get_s(int(self.mask_batch_size / 2))
                    m = get_random_masks_different_s(half_mask_shape, s_per_mask, dtype='uint')
                    m2 = np.apply_along_axis(increment_mask, 1, m, self.flip_size)
                    m = np.concatenate([m, m2], axis=0)
                    x, m, y = create_masked_batch(x, m, y)  # creating masked batch
                    yield [x, m], y

        tb = TensorBoard(
            log_dir=os.path.join(FIDL_TENSORBOARD_LOG_DIR, self.dataset_name, self.datetime_start + "_task"),
            write_graph=True, update_freq="epoch", profile_batch=2)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=lr_patience, min_delta=0.0001)
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=early_stop, restore_best_weights=True)
        model_ckpt = ModelCheckpoint(os.path.join(FIDL_MODEL_DIR, self.dataset_name, self.datetime_start + "task.hdf5"),
                                     save_best_only=True, monitor='val_loss')
        callbacks = [tb, reduce_lr, early_stop, model_ckpt]
        h = self.task_model.fit(x=data_gen("train"), epochs=self.task_model_current_epoch+max_epoch,
                                callbacks=callbacks, validation_data=data_gen("val"),
                                steps_per_epoch=self.train_length, validation_steps=self.val_length,
                                validation_freq=val_freq, max_queue_size=1, initial_epoch = self.task_model_current_epoch)
        self.task_model_current_epoch += len(h.history['loss'])
        logging.debug("Task model finished at epoch number: "+str(self.task_model_current_epoch))
        return

    def fit_phase_2(self, max_epoch, early_stop, lr_patience, n_pfs_samples_in_memory,
                    n_task_samples_in_memory=None, mask_batches_per_epoch=32,
                    reuse_samples=True):
        # - 2nd PHASE
        # - CREATE DATA FOR PFS TRAINING
        self.save_pfs_data = True
        self.task_mode = self.mode_dict[self.PFS_target_mode_str]

        def data_gen():
            logging.debug("Init datagen for phase 2 with task network target mode: " + str(self.PFS_target_mode_id))
            half_mask_shape = (int(self.mask_batch_size / 2),) + self.data_shape
            logging.debug("Half Mask shape: " + str(half_mask_shape))

            batch_index = 0
            self.dataImporter.on_epoch_start()
            while True:
                # clear data
                #logging.debug("Clear cached data. Entering data creation loop.")
                self.clear_PFS_data()
                #tracemalloc.start(25)
                #snapshot1 = tracemalloc.take_snapshot()
                #logging.debug("Enter data gathering loop.")
                while True:
                    #logging.debug("Data gathering batch id: " + str(batch_index))
                    if batch_index >= self.length_dict[self.PFS_target_mode_str]:
                        #logging.debug("Resetting the data gathering epoch.")
                        batch_index = 0  # reset the task data epoch
                    if len(self.PFS_X) * self.mask_batch_size >= n_pfs_samples_in_memory:
                        #logging.debug("Breaking out of data gathering loop.")
                        break  # start to yield the data
                    x, y = self.dataImporter.get_batch(self.PFS_target_mode_str, batch_index)
                    s_per_mask = self.mask_optimizer.get_s(int(self.mask_batch_size / 2))
                    x_batch_size = x.shape[0]
                    m = get_random_masks_different_s(half_mask_shape, s_per_mask, dtype='uint')
                    m2 = np.apply_along_axis(increment_mask, 1, m, self.flip_size)
                    m = np.concatenate([m, m2], axis=0)
                    x, m, y = create_masked_batch(x, m, y)  # creating masked batch
                    self.task_model.evaluate(x=[x, m], y=y, batch_size=self.batch_size, verbose=0)
                    self.append_PFS_data(m,x_batch_size)
                    batch_index += 1
                #concatenate data from list -> np array
                self.PFS_X = np.concatenate(self.PFS_X,axis=0)
                self.PFS_y_true = np.concatenate(self.PFS_y_true, axis=0)


                # get m_opt
                #snapshot2 = tracemalloc.take_snapshot()

                #logging.info("\n")
                #logging.info( "tracemalloc")
                #for diff in snapshot2.compare_to(snapshot1, "lineno")[:10]:
                #        logging.debug(str(diff)[:])
                #snapshot1 = tracemalloc.take_snapshot()

                #logging.debug("Entering the yielding loop that reuses data, samples: "+str(len(self.PFS_X)))
                while True:  # yield always the same data, never leave this loop
                    pfs_batch_index = 0
                    # logging.debug("New data yielding epoch.")
                    while True:
                        start_idx = pfs_batch_index * self.mask_batch_size
                        end_idx = start_idx + self.mask_batch_size
                        if start_idx >= len(self.PFS_X):
                            # logging.debug("Breaking out from the loop.")
                            break

                        yield [self.PFS_X[start_idx:end_idx],
                               self.PFS_y_true[start_idx:end_idx]], np.empty(self.mask_batch_size)
                        pfs_batch_index += 1
                    if reuse_samples == False:
                        break


        # - FIT PFS NN UNTIL TRAINING LOSS STOPS INCREASING
        tb = TensorBoard(
            log_dir=os.path.join(FIDL_TENSORBOARD_LOG_DIR, self.dataset_name, self.datetime_start + "_pfs"),
            write_graph=True, update_freq="epoch", profile_batch=2)
        reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=lr_patience, min_delta=0.0001)
        early_stop = EarlyStopping(monitor="loss", min_delta=0, patience=early_stop, restore_best_weights=True)
        model_ckpt = ModelCheckpoint(os.path.join(FIDL_MODEL_DIR, self.dataset_name, self.datetime_start + "pfs.hdf5"),
                                     save_best_only=True, monitor='loss')
        callbacks = [tb, reduce_lr, early_stop, model_ckpt]
        h = self.PFS_model.fit(x=data_gen(),
                               epochs=self.PFS_model_current_epoch+max_epoch,
                               callbacks=callbacks,
                               steps_per_epoch=mask_batches_per_epoch,
                               max_queue_size=1,
                               initial_epoch = self.PFS_model_current_epoch)
        self.PFS_model_current_epoch += len(h.history['loss'])
        logging.debug("Task model finished at epoch number: " + str(self.PFS_model_current_epoch))
        self.clear_PFS_data()
        return

    def fit_phase_3(self,m_opt_period, operator_train_period, operator_val_period,
                    search_train_period,max_counter = 10000, m_opt_starting_iterations=10):

        def search_net_gen():
            while True:
                if (self.new_pfs_data_present==True):
                    l = min(len(self.PFS_X),len(self.PFS_y_true))
                    logging.debug("Separating PFS data lists of legnth: "+str(l))
                    X = np.concatenate(self.PFS_X[:l],axis=0)
                    y_true = np.concatenate(self.PFS_y_true[:l],axis=0)
                    self.PFS_X = []#self.PFS_X[l:]
                    self.PFS_y_true = []#self.PFS_y_true[l:]
                    self.new_pfs_data_present = False
                    gc.collect()
                length = int(len(X) / self.mask_batch_size)
                logging.debug("Begin search net data loop. Samples X/y:" + str(len(X))+" "+str(len(y_true)))

                for i in range(length):
                    #logging.debug("Getting search net batch id:"+str(i))
                    start_idx = i * self.mask_batch_size
                    end_idx = start_idx + self.mask_batch_size
                    yield [X[start_idx:end_idx],y_true[start_idx:end_idx]], \
                          np.empty(self.mask_batch_size)

        def operator_net_gen(mode):
            while True:
                self.dataImporter.on_epoch_start()
                for i in range(self.length_dict[mode]):
                    #logging.debug("Getting operator data batch id:"+str(i))
                    x, y = self.dataImporter.get_batch(mode, i)
                    m = get_mask_batch(m_opt)
                    if(self.PFS_target_mode_str == mode):
                        self.PFS_X.append(np.copy(m))
                    x, m, y = create_masked_batch(x, m, y)  # creating masked batch
                    #self.last_used_m = m
                    yield [x, m], y
        half_mask_shape = (int(self.mask_batch_size / 2),) + self.data_shape
        quarter_mask_shape = (int(self.mask_batch_size / 4),) + self.data_shape

        def get_mask_batch(m_opt_):
            m = np.empty(((self.mask_batch_size,) + self.data_shape))
            mean_s = max(m_opt_sum,self.flip_size*5)
            sampled_s = self.mask_optimizer.get_s(n_samples=half_mask_shape[0] - quarter_mask_shape[0],mean=int(mean_s))
            m[0] = m_opt_
            for i in range(1, quarter_mask_shape[0]):
                m[i] = MaskOptimizer.flip_masks(np.copy(m_opt_)[None,:], self.flip_size, with_repetitions=True)

            m[quarter_mask_shape[0]:half_mask_shape[0]] = get_random_masks_different_s(
                (half_mask_shape[0] - quarter_mask_shape[0],) + self.data_shape, sampled_s, dtype='uint')
            m[half_mask_shape[0]:] = np.apply_along_axis(increment_mask, 1, m[:half_mask_shape[0]], self.flip_size)
            #print("Mask batch debug:")
            #for idx,i in enumerate(m):
            #    print(idx, np.nonzero(i)[0])
            return m.astype('float')

        operator_training_finished = False
        m_opt = np.zeros(self.data_shape)
        m_opt_sum = 0
        operator_tb = TensorBoard(log_dir=os.path.join(FIDL_TENSORBOARD_LOG_DIR, self.dataset_name, self.datetime_start + "_task"),
                                        write_graph=True, update_freq="epoch", profile_batch=0)
        search_tb = TensorBoard(log_dir=os.path.join(FIDL_TENSORBOARD_LOG_DIR, self.dataset_name, self.datetime_start + "_pfs"),
                                        write_graph=True, update_freq="epoch", profile_batch=0)
        #self.operator_callbacks=[DataSavingCallback(self),CleaningCallback()]
        self.operator_callbacks = [operator_tb,DataSavingCallback(self, operator_tb),CleaningCallback()]
        #self.operator_callbacks.append()
        self.search_callbacks = [search_tb, CleaningCallback()]

        search_gen = search_net_gen()
        operator_train_gen = operator_net_gen("train")
        operator_val_gen = operator_net_gen("val")

        # - start with several m_opt iterations:
        for i in range(m_opt_starting_iterations):
            imps = self.PFS_model.predict(m_opt[None, :])
            m_opt = self.mask_optimizer.advance_mask(imps[0], m_opt, self.flip_size)
            m_opt_sum = np.sum(m_opt)
        logging.debug("Started with m_opt with size: " + str(m_opt_sum))
        # - 3rd PHASE
        tracemalloc.start(25)
        snapshot1 = tracemalloc.take_snapshot()
        for counter in range(max_counter):
            # get m_opt
            snapshot2 = tracemalloc.take_snapshot()

            logging.info("\n")
            logging.info(str(counter)+"tracemalloc")
            for diff in snapshot2.compare_to(snapshot1, "lineno")[:10]:
                if(counter>0):
                    logging.debug(str(diff)[:])
            snapshot1 = tracemalloc.take_snapshot()
            if(counter % m_opt_period == 0):
                imps = self.PFS_model.predict(m_opt[None, :])
                m_opt = self.mask_optimizer.advance_mask(imps[0], m_opt, self.flip_size)
                m_opt_sum = np.sum(m_opt)
                logging.debug(str(counter)+": created a new m_opt with size: "+str(m_opt_sum)+" and:"+str(np.nonzero(m_opt)))
            # train operator net
            if (counter % operator_train_period == 0 and operator_training_finished==False):
                self.task_mode = self.mode_dict["train"]
                logging.debug(str(counter)+": Start training epoch of the  operator.")
                #clear data if mode is train
                #for i in range(self.train_length):
                #    next(operator_train_gen)
                h = self.task_model.fit(x=operator_train_gen,
                                        epochs=self.task_model_current_epoch+1,
                                        callbacks=self.operator_callbacks,
                                        steps_per_epoch=self.train_length,
                                        max_queue_size=1,
                                        initial_epoch = self.task_model_current_epoch)
                self.task_model_current_epoch += len(h.history['loss'])

            # validate operator net
            if (counter % operator_val_period == 0):
                self.task_mode = self.mode_dict["val"]
                logging.debug(str(counter) + ": Start val epoch of the  operator.")

                #for i in range(self.val_length):
                #    x,y = next(operator_val_gen)
                #    #self.task_model.evaluate(x=x,y=y,verbose=0)
                #    self.task_model.test_on_batch(x=x,y=y)
                #    self.operator_callbacks[0].on_test_batch_end(i)
                self.task_model.evaluate(x=operator_val_gen,
                                             callbacks=self.operator_callbacks,
                                             steps=self.val_length)
            # train search net
            if (counter % search_train_period == 0):
                logging.debug(str(counter) + ": Start training epoch of the search net.")
                #for i in range(int(len(self.PFS_X) )):
                #    next(search_gen)
                steps = int(len(self.PFS_X) )
                h = self.PFS_model.fit(x=search_gen,
                                       epochs=self.PFS_model_current_epoch+1,
                                       callbacks=self.search_callbacks,
                                       steps_per_epoch=steps,
                                       max_queue_size=1,
                                       initial_epoch = self.PFS_model_current_epoch)
                self.PFS_model_current_epoch += len(h.history['loss'])
            #if counter % 40 == 0 and counter >0:
            #    top_stat = snapshot2.statistics('traceback')
            #    for stat in top_stat[:10]:
            #        logging.debug("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
            #        for line in stat.traceback.format():
            #            logging.debug(line)
            #    return


        return

    def loadPFSModel(self, PFS_model):
        self.PFS_inputs = PFS_model.inputs
        self.PFS_output = PFS_model.outputs
        # logging.debug("PFS model input shape:"+str([input.shape for input in self.PFS_inputs]))
        # logging.debug("PFS model ouput shape:" + str([output.shape for output in self.PFS_output]))
        logging.debug("PFS  model inputs: " + str(self.PFS_inputs))
        logging.debug("PFS  model output: " + str(self.PFS_output))
        self.PFS_model = PFS_model

    def loadTaskModel(self, task_model):
        self.task_inputs = task_model.inputs
        self.task_output = task_model.outputs
        # logging.debug("Task model input shape:" + str([input.shape for input in self.task_inputs]))
        # logging.debug("Task model ouput shape:" + str([output.shape for output in self.task_output]))
        logging.debug("Task model inputs: " + str(self.task_inputs))
        logging.debug("Task model output: " + str(self.task_output))
        self.task_model = task_model

    def compileTaskModel(self, loss, metrics, PFS_target_ID, PFS_target_mode='val'):
        """
        :param PFS_target_ID: -1 for task_model loss, 0 and above for all the metrics
        :PFS_target_mode: "val" or "train"
        """

        def get_record_function(func):
            def record_func(y_true, y_pred):
                # y_true = K.print_tensor(y_true, message="true")
                # y_pred = K.print_tensor(y_pred, message="pred")
                res = func(y_true, y_pred)
                self.task_output_vals.assign(res)
                # res = K.print_tensor(res,message="loss")
                return res

            return record_func

        assert PFS_target_ID >= -1 and PFS_target_ID < len(metrics)
        self.PFS_target_mode_id = self.mode_dict[PFS_target_mode]
        self.PFS_target_mode_str = PFS_target_mode
        task_model_loss = tf.function(func=get_record_function(loss)) if PFS_target_ID == -1 else loss
        if (PFS_target_ID >= 0):
            metrics[PFS_target_ID] = tf.function(func=get_record_function(metrics[PFS_target_ID]))
        self.task_model.compile(optimizer="nadam", loss=task_model_loss, metrics=metrics, run_eagerly=self.eager_mode)

    def compilePFSModel(self, metrics=[], use_verbose_loss=False):
        """
        """

        # pfs_input = self.PFS_inputs[0]
        def get_loss_function():
            def loss_func_verbose(y_true, y_pred, pfs_input):
                y_true = K.print_tensor(y_true, message="true")
                y_pred = K.print_tensor(y_pred, message="pred")
                # Split masks, find differences
                base_masks, perturbed_masks = tf.split(pfs_input, num_or_size_splits=2, axis=0)
                base_masks = K.print_tensor(base_masks, message="base_masks")
                perturbed_masks = K.print_tensor(perturbed_masks, message="perturbed_masks")
                affected_features = tf.abs(perturbed_masks - base_masks)
                affected_features = K.print_tensor(affected_features, message="affected_features")

                # Split predicted y, get difference for unmasked inputs, sum over all inputs
                y_pred_base, y_pred_perturbed = tf.split(y_pred, num_or_size_splits=2, axis=0)
                y_pred_base = K.print_tensor(y_pred_base, message="y_pred_base")
                y_pred_perturbed = K.print_tensor(y_pred_perturbed, message="y_pred_perturbed")
                y_pred_diff = y_pred_perturbed - y_pred_base
                y_pred_diff = K.print_tensor(y_pred_diff, message="y_pred_diff_1")
                y_pred_diff = y_pred_diff * affected_features
                y_pred_diff = K.print_tensor(y_pred_diff, message="y_pred_diff_2")
                y_pred_diff = tf.reduce_sum(y_pred_diff, axis=1, keepdims=True)
                y_pred_diff = K.print_tensor(y_pred_diff, message="y_pred_diff_3")
                full_y_pred = tf.concat([y_pred_diff, -y_pred_diff], axis=0)
                full_y_pred = K.print_tensor(full_y_pred, message="full_y_pred")

                # Split target y, get differences for perturbed/unperturbed masks
                y_true_base, y_true_perturbed = tf.split(y_true, num_or_size_splits=2, axis=0)
                y_true_diff = y_true_perturbed - y_true_base
                y_true_diff = K.print_tensor(y_true_diff, message="y_true_diff_1")
                full_y_true = tf.concat([y_true_diff, -y_true_diff], axis=0)
                full_y_true = K.print_tensor(full_y_true, message="full_y_true")

                # Calculate MSE loss
                loss = mean_squared_error(full_y_true, full_y_pred)
                loss = K.print_tensor(loss, message="loss")
                return loss

            def loss_func(y_true, y_pred, pfs_input):
                # pfs_input = self.PFS_inputs[0]
                base_masks, perturbed_masks = tf.split(pfs_input, num_or_size_splits=2, axis=0)
                affected_features = tf.abs(perturbed_masks - base_masks)

                # OLD, incorrect implementation - results randommly offset
                #y_pred_base, y_pred_perturbed = tf.split(y_pred, num_or_size_splits=2, axis=0)
                #y_pred_diff = y_pred_perturbed - y_pred_base
                #y_pred_diff = y_pred_diff * affected_features
                #y_pred_diff = tf.math.reduce_sum(y_pred_diff, axis=1, keepdims=True)
                #full_y_pred = tf.concat([y_pred_diff, -y_pred_diff], axis=0)

                # NEW
                affected_features = tf.concat([affected_features, affected_features], axis=0)
                masked_y_pred = y_pred * affected_features
                full_y_pred = tf.math.reduce_sum(masked_y_pred, axis=1, keepdims=True)

                y_true_base, y_true_perturbed = tf.split(y_true, num_or_size_splits=2, axis=0)
                y_true_diff = y_true_perturbed - y_true_base
                full_y_true = tf.concat([y_true_diff, -y_true_diff], axis=0)

                loss = mean_squared_error(full_y_true, full_y_pred)
                return loss

            if (use_verbose_loss == True):
                logging.debug("Using verbose loss for PFS.")
            return loss_func_verbose if use_verbose_loss == True else loss_func

        y_true_input = Input((1,))
        mask_input = Input(self.data_shape)
        new_output = self.PFS_model(mask_input)

        self.PFS_model = Model(inputs=[mask_input, y_true_input],
                               outputs=new_output, name="PFS_model_env")

        self.PFS_model.add_loss(get_loss_function()(y_true_input, new_output, mask_input))
        self.PFS_model.compile(metrics=metrics, run_eagerly=self.eager_mode)
        # self.PFS_model.compile(loss=tf.function(func=get_loss_function()),
        #                       metrics=metrics,run_eagerly=self.eager_mode)

    def train_batch_task(self, x, m, y):
        self.task_mode = self.mode_dict["train"]
        x_masked, m_expanded, y_expanded = create_masked_batch(x, m, y)
        self.task_hist.append(self.task_model.train_on_batch([x_masked, m_expanded], y_expanded))
        self.append_PFS_data(m)
        print("train: ", K.eval(self.task_output_vals))

    def test_batch_task(self, x, m, y):
        self.task_mode = self.mode_dict["test"]
        x_masked, m_expanded, y_expanded = create_masked_batch(x, m, y)
        self.task_hist.append(self.task_model.test_on_batch([x_masked, m_expanded], y_expanded))
        self.append_PFS_data(m)
        print("test: ", K.eval(self.task_output_vals))

    def validate_batch_task(self, x, m, y):
        self.task_mode = self.mode_dict["val"]
        x_masked, m_expanded, y_expanded = create_masked_batch(x, m, y)
        self.task_hist.append(self.task_model.test_on_batch([x_masked, m_expanded], y_expanded))
        self.append_PFS_data(m)
        print("val: ", K.eval(self.task_output_vals))

    def append_PFS_data(self, m, used_data_batch_size):
        #logging.debug("Appending data. Constants: " + str(self.save_pfs_data) + " " + str(self.task_mode))
        data_batch_size = int(m.shape[0]/self.mask_batch_size)

        if self.save_pfs_data == True:
            if self.task_mode == self.PFS_target_mode_id:
                y = K.eval(self.task_output_vals)
                #print(y.shape,m.shape)
                assert len(y) == len(m)
                # average the output
                y = np.mean(reshape_masked_variable(y, self.mask_batch_size,data_batch_size), axis=0)
                # choose all the masks
                m = reshape_masked_variable(m, self.mask_batch_size, data_batch_size)[0]
                #logging.debug("Appending data of shapes m:" + str(m.shape) + " y:" + str(y.shape))
                self.PFS_y_true.append(y)
                self.PFS_X.append(m)

    def clear_PFS_data(self):
        self.PFS_y_true=[]
        self.PFS_X=[]
