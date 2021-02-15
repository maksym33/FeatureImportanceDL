import logging
import numpy as np
import gc
from .config import FIDL_RS
from tensorflow import keras
from tensorflow.keras import backend as K


class CleaningCallback(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        logging.debug("Cleaning callback.")
        gc.collect()
        K.clear_session()
        return

    def on_test_end(self, logs=None):
        logging.debug("Cleaning callback.")
        gc.collect()
        K.clear_session()
        return

class DataSavingCallback(keras.callbacks.Callback):
    def __init__(self, model_trainer,tensorboard_cb=None):
        self.model_trainer = model_trainer
        self.tensorboard_cb = tensorboard_cb

    def on_train_begin(self, logs=None):
        #if self.model_trainer.PFS_target_mode_str == "train" and self.model_trainer.new_pfs_data_present == False:
        #    self.model_trainer.clear_PFS_data()
        return

    def on_test_begin(self, logs=None):
        #if self.model_trainer.PFS_target_mode_str == "val" and self.model_trainer.new_pfs_data_present == False:
        #    self.model_trainer.clear_PFS_data()
        return
    def on_test_end(self, logs=None):
        if not(self.tensorboard_cb is None):
            new_logs={}
            for key in logs.keys():
                new_logs['val_'+key]=logs[key]
            self.tensorboard_cb.on_epoch_end(
                self.model_trainer.task_model_current_epoch,new_logs)
        return

    def on_train_batch_end(self, batch, logs=None):
        if self.model_trainer.PFS_target_mode_str == "train":
            self.append_data()
            #self.model_trainer.append_PFS_data(self.model_trainer.last_used_m, self.model_trainer.data_batch_size)
        return

    def on_test_batch_end(self, batch, logs=None):
        if self.model_trainer.PFS_target_mode_str == "val":
            self.append_data()
            #self.model_trainer.append_PFS_data(self.model_trainer.last_used_m, self.model_trainer.data_batch_size)
        return

    def append_data(self):
        y = K.eval(self.model_trainer.task_output_vals)
        data_batch_size = int(y.shape[0]/self.model_trainer.mask_batch_size)
        y = np.mean(reshape_masked_variable(y, self.model_trainer.mask_batch_size, data_batch_size), axis=0)
        self.model_trainer.PFS_y_true.append(y)
        self.model_trainer.new_pfs_data_present = True
        #logging.debug("Correponding lenghts after appending: y/X: "+str(len(self.model_trainer.PFS_X))+" "+str(len(self.model_trainer.PFS_y_true)))

def create_masked_batch(x, m, y):
    """
    x =     [[1,2],[3,4]]       -> [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]]
    x_prim=     (MASKING)       -> [[0,0],[1,0],[1,2],[0,0],[3,0],[3,4]]
    m_prim= [[0,0],[1,0],[1,1]] -> [[0,0],[1,0],[1,1],[0,0],[1,0],[1,1]]
    y_prim= [1,3]               -> [1    ,1    ,1    ,3    ,3    ,3    ]
    """
    x_prim = np.repeat(x, len(m), axis=0)
    y_prim = np.repeat(y, len(m), axis=0)
    m_prim = np.tile(m, (len(x), 1)).astype('float32') # change type int->float
    x_prim = np.where(m_prim, x_prim, 0)  # MASKING
    return x_prim, m_prim, y_prim

def reshape_masked_variable(v, mask_batch_size,data_batch_size):
    """
    inverts the reshaping done in the create_masked_batch function
    returns shape (data_bs_shape,mask_bs_shape,...)
    """
    v_prim = np.reshape(v,(data_batch_size,mask_batch_size)+v.shape[1:])
    return v_prim



# noinspection PyUnreachableCode
def increment_mask(mask, n, return_changed_idx=False):
    if __debug__:
        assert np.count_nonzero(mask) + n <= mask.size
        assert n >= 0
    new_mask = np.copy(mask)
    changed_idx = FIDL_RS.choice(np.squeeze(np.argwhere(mask == 0)), n, replace=False)
    new_mask[changed_idx] = 1
    if return_changed_idx == True:
        return new_mask, changed_idx
    else:
        return new_mask


# noinspection PyUnreachableCode
def decrement_mask(mask, n, return_changed_idx=False):
    if __debug__:
        assert np.count_nonzero(mask) >= n
        assert n >= 0
    new_mask = np.copy(mask)
    changed_idx = FIDL_RS.choice(np.squeeze(np.nonzero(mask)[0]), n, replace=False)
    new_mask[changed_idx] = 0
    if return_changed_idx == True:
        return new_mask, changed_idx
    else:
        return new_mask




# noinspection PyUnreachableCode
def get_random_masks_same_s(shape, s_per_mask, dtype='uint'):
    """
    :param s_per_mask: number of ones in a mask
    :param shape:
    :return: can return masks with bigger size (more 1s) as this is a random function
    """
    if __debug__:
        assert len(shape) == 2
        assert s_per_mask <= shape[1]
    return np.apply_along_axis(increment_mask, 1, np.zeros(shape=shape, dtype=dtype), s_per_mask)

# noinspection PyUnreachableCode
def get_random_masks_different_s(shape, s_per_mask, dtype='uint'):
    """
    :param s_per_mask: number of ones in a mask
    :param shape:
    :return: can return masks with bigger size (more 1s) as this is a random function
    """
    if __debug__:
        assert len(shape) == 2
        assert len(s_per_mask) == shape[0]
    m = np.empty(shape=shape, dtype=dtype)
    m_i = np.zeros(shape=shape[1:], dtype=dtype)
    for i in range(shape[0]):
        m[i] = increment_mask(m_i,s_per_mask[i])
    return m
    #return np.apply_along_axis(increment_mask, 1, np.zeros(shape=shape, dtype=dtype), s_per_mask)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


