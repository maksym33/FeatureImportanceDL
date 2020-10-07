import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Layer, multiply, Concatenate,Flatten
MISSING_INPUT_CONSTANT = -10
IGNORE_OUTPUT_CONSTANT = -5
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

class SubjectNetwork():
    def __init__(self,mask_batch_size,uses_label=False,tensorboard_logs_dir=""):        
        self.batch_size = mask_batch_size
        self.mask_batch_size = mask_batch_size
        self.tr_loss_history = []
        self.te_loss_history = []
        self.y_pred_std_history = []
        self.y_true_std_history = []
        self.tf_logs = tensorboard_logs_dir
        self.epoch_counter = 0        
        self.uses_label = uses_label
        self.data_masks = None
        self.data_targets = None
        self.best_performing_mask = None
        self.sample_weights = None
    def set_label_input_params(self,y_shape,y_input_layer):
        self.label_input_layer =y_input_layer
        self.label_shape = y_shape
    def create_dense_model(self,input_shape,dense_arch):
        input_mask_layer = Input(shape=input_shape)
        if(self.uses_label==True):       
            input_label = Input(shape=self.label_shape)
        x = Flatten()(input_mask_layer)
        for i in range(len(dense_arch[:-1])):
            x = Dense(dense_arch[i],activation="sigmoid")(x)
            if(self.uses_label == True and self.label_input_layer == i):
                print("Concatenating additional input layer!")
                x = K.concatenate([x,input_label])
        x = Dense(dense_arch[-1],activation="linear")(x)
        if(self.uses_label==True):
            self.model = Model(inputs=[input_mask_layer,input_label],outputs=x)
        else:    
            self.model = Model(inputs=[input_mask_layer],outputs=x)
        print("Subject Network model built:")
        self.model.summary()
    def named_logs(self,model, logs):
        result = {}
        try:
            iterator = iter(logs)
        except TypeError:
            logs=[logs]
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result
    def compile_model(self):
        self.model.compile(loss='mae',optimizer='adam',metrics=[self.get_y_std_metric(True),self.get_y_std_metric(False)])       
        if(self.tf_logs != ""):
            log_path = './logs'
            self.tb_clbk = TensorBoard(self.tf_logs)
            self.tb_clbk.set_model(self.model)        
    def train_one(self,x,y):        
        curr_loss = self.model.train_on_batch(x=x,y=y)
        self.tr_loss_history.append(curr_loss)
        self.epoch_counter += 1
        if(self.tf_logs != ""):
            self.tb_clbk.on_epoch_end(self.epoch_counter,self.named_logs(self.model,curr_loss))   
    def train_one(self,epoch_number,apply_weights): # train on data in object memory        
        if(apply_weights==False):
            curr_loss = self.model.train_on_batch(x=self.data_masks,y=self.data_targets)
        else:
            curr_loss = self.model.train_on_batch(x=self.data_masks,y=self.data_targets,sample_weight = self.sample_weights)
        self.best_performing_mask = self.data_masks[np.argmin(self.data_targets,axis=0)]        
        self.tr_loss_history.append(curr_loss)
        self.epoch_counter= epoch_number
        if(self.tf_logs != ""):
            self.tb_clbk.on_epoch_end(self.epoch_counter,self.named_logs(self.model,curr_loss))       
        self.data_masks = None
        self.data_targets=None
        
        if(self.uses_label==True):
            self.data_obj_labels = None
    def append_data(self,x,y):     
        if(self.uses_label==False):
            if(self.data_masks is None):
                self.data_masks = x
                self.data_targets = y
            else:         
                self.data_masks = np.concatenate([self.data_masks,x],axis=0)
                self.data_targets = tf.concat([self.data_targets,y],axis=0)
        else:
            if(self.data_masks is None):
                self.data_masks = x[0]
                self.data_obj_labels = x[1]
                self.data_targets = y
            else:         
                self.data_masks = np.concatenate([self.data_masks,x[0]],axis=0)
                self.data_obj_labels = np.concatenate([self.data_obj_labels,x[1]],axis=0)     
                self.data_targets = tf.concat([self.data_targets,y],axis=0)   
    def test_one(self,x,y):        
        y_pred = self.model.predict(x=x)
        curr_loss = self.model.test_on_batch(x=x,y=y)
        self.te_loss_history.append(curr_loss)
        #print("SN test loss: "+str(curr_loss))  
        #print("SN prediction: "+str(np.squeeze(curr_loss))) 
        #print("SN targets: "+str(np.squeeze(y_pred))) 
        return curr_loss
    def predict(self,x):        
        y_pred = self.model.predict(x=x)
        return y_pred
    def get_y_std_metric(self,ifpred=True):
        def y_pred_std_metric(y_true,y_pred):
            y_pred_std = K.std(y_pred)
            self.y_pred_std_history.append(y_pred_std)
            return y_pred_std
        def y_true_std_metric(y_true,y_pred):
            y_true_std = K.std(y_true)
            self.y_true_std_history.append(y_true_std)
            return y_true_std
        if(ifpred==True):
            return y_pred_std_metric
        else:
            return y_true_std_metric
    def plot_loss_hist(self,ifLogScale=False,PlotAverageOfNPoints = 1):
        def movingaverage(interval, window_size):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'valid')
        data = self.tr_loss_history
        if(PlotAverageOfNPoints>1):
             data = movingaverage(data, PlotAverageOfNPoints)   
        x = np.arange(len(data))+PlotAverageOfNPoints
        plt.plot(x,data)
        if(ifLogScale==True):
            plt.yscale('log')
        plt.show()
