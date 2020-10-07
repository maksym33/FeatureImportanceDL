
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from scipy.spatial.distance import pdist, squareform
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Layer, multiply, Concatenate,Flatten
MISSING_INPUT_CONSTANT = -10
IGNORE_OUTPUT_CONSTANT = -5
from tensorflow.keras import backend as K

from DataGenerator import UniformDataGenerator
from SubNet import SubjectNetwork
from ObjNet import ObjectNetwork
from MaskOptimizer import MaskOptimizer
import os,datetime,time
logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)

import sys
def mean_squared_error(y_true,y_pred):
    return K.mean((y_true-y_pred)*(y_true-y_pred),axis=1)
def keras_accuracy(y_true,y_pred):
    return K.categorical_crossentropy(y_true,y_pred)
def keras_mean(losses):
    return K.mean(losses)
def keras_mean_ax_1(losses):
    return K.mean(losses,axis=1)
def tf_mean_ax_0(losses):
    return tf.reduce_mean(losses,axis=0)
def tf_mean_ax_1(losses):
    return tf.reduce_mean(losses,axis=1)
def numpy_mean(losses):
    return np.mean(losses)
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

class FeatureSelector():
    def __init__(self,data_shape,unmasked_data_size,data_batch_size,mask_batch_size,str_id = "",epoch_on_which_subnet_trained=8):
        self.data_shape = data_shape
        self.data_size = np.zeros(data_shape).size
        self.unmasked_data_size = unmasked_data_size
        self.logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%m%d-%H%M%S"))
        self.data_batch_size = data_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_batch_size = mask_batch_size * data_batch_size
        self.str_id = str_id
        self.prev_mopt_condition = False
        self.epoch_on_which_subnet_trained = epoch_on_which_subnet_trained
        
        self.time_subnet_train = 0

        self.time_prepare_m_opt =0
        
    def clear_networks():
        K.clear_session()
    def create_dense_objnet(self,arch, activation, metrics=None,error_func=mean_squared_error,use_masks=0,es_patience=800):
        self.objnet = ObjectNetwork(self.data_batch_size,self.mask_batch_size,self.logdir+"objnet"+self.str_id,use_masks=use_masks)
        print("Creating objnet model")
        self.objnet.create_dense_model(self.data_shape,arch,activation)
        print("Compiling objnet")
        #self.objnet_compile_args = [mean_squared_error,keras_mean,numpy_mean,metrics]
        self.objnet.compile_model(error_func,tf.reduce_mean,tf_mean_ax_0,metrics)
        print("Created objnet")
    def create_conv_objnet(self,filters,kernels,dense_arch,activation,img_shape=None,channels=1, padding="same", metrics=None,error_func=None,es_patience=800):
        self.objnet = ObjectNetwork(self.data_batch_size,self.mask_batch_size,self.logdir+"objnet"+self.str_id,use_masks=0)
        print("Creating objnet model")
        if(channels==1):
            self.objnet.create_1ch_conv_model(self.data_shape,image_shape=img_shape,filter_sizes=filters, kernel_sizes=kernels,dense_arch=dense_arch,padding=padding,last_activation=activation)
        else:
            self.objnet.create_2ch_conv_model(self.data_shape,image_shape=img_shape,filter_sizes=filters, kernel_sizes=kernels,dense_arch=dense_arch,padding=padding,last_activation=activation)
        print("Compiling objnet")
        self.objnet.compile_model(error_func,tf.reduce_mean,tf_mean_ax_0,metrics)
        print("Created objnet")
    def create_dense_subnet(self,arch):
        self.subnet = SubjectNetwork(self.mask_batch_size,tensorboard_logs_dir=self.logdir+"subnet_"+self.str_id)
        self.subnet.create_dense_model(self.data_shape,arch)
        self.subnet.compile_model()
    def create_mask_optimizer(self,epoch_condition=5000, maximize_error=False,record_best_masks=False,perturbation_size=2,use_new_optimization=False):
        self.mopt = MaskOptimizer(self.mask_batch_size,self.data_shape,self.unmasked_data_size,epoch_condition=epoch_condition, record_best_masks = record_best_masks,perturbation_size = perturbation_size,maximize_error=maximize_error,use_new_optimization=use_new_optimization)
        self.subnet.sample_weights = self.mopt.get_mask_weights(self.epoch_on_which_subnet_trained)
#    def create_data_generator(self,gen_func):  
#        self.udg = UniformDataGenerator(self.data_batch_size,self.mask_batch_size, #self.data_shape,gen_func,self.unmasked_data_size)

    def test_networks_on_data(self,x,y,masks):
            #x,y = self.udg.get_batch(number_of_data_batches)
            m = masks
            x_prim,m_prim,y_prim, losses = self.objnet.test_one(x,m,y)
            target_shape = (len(y),len(masks))
            losses = self.objnet.get_per_mask_loss(target_shape)  
            print("SN targets: "+str(losses))
            #print("SN mean targets: "+str(np.mean(losses,axis=0)))
            sn_preds = np.squeeze(self.subnet.predict(m))
            print("SN preds: "+str(sn_preds))            
    def train_networks_on_data(self,x_tr,y_tr,number_of_batches,val_data=None,val_freq=16,lr=1,iters=10):
        use_val_data = True
        if val_data is None:
            use_val_data = False
        X_val=None
        y_val=None
        if(use_val_data==True):
            X_val = val_data[0]
            y_val = val_data[1]
            
        for i in progressbar(range(number_of_batches), "Training batch: ", 50):
            mopt_condition = self.mopt.check_condiditon()
                    
            random_indices = np.random.randint(0,len(x_tr),self.data_batch_size)
            x = x_tr[random_indices,:]
            y = y_tr[random_indices]
            subnet_train_condition = ((self.objnet.epoch_counter % self.epoch_on_which_subnet_trained) == 0)
            t0 = time.time()
            m = self.mopt.get_new_mask_batch(self.subnet.model,self.subnet.best_performing_mask, iters=iters,lr=lr,gen_new_opt_mask=subnet_train_condition )
            self.time_prepare_m_opt += (time.time() - t0)
            
            self.objnet.train_one(x,m,y)
            losses = self.objnet.get_per_mask_loss()
            losses = losses.numpy()
            self.subnet.append_data(m,losses)
            t0 = time.time()
            if(subnet_train_condition):
                self.subnet.train_one(self.objnet.epoch_counter,mopt_condition)
            self.time_subnet_train += (time.time() - t0)

            self.prev_mopt_condition = mopt_condition
            if(use_val_data == True and self.objnet.epoch_counter % val_freq==0 ):          
                self.objnet.test_one(X_val,m,y_val)
            if(self.objnet.useEarlyStopping==True and self.objnet.ES_stop_training==True):
                print("Activate early stopping at training epoch/batch: "+str(self.objnet.epoch_counter))
                print("Loading weights from epoch: "+str(self.objnet.ES_best_epoch))
                self.objnet.model.set_weights(self.objnet.ES_best_weights)
                break
                
                
    def show_mask_history(self,ifraw=True,smoothing_points = 1):
        def movingaverage(interval, window_size):
            window = np.ones(int(window_size))/float(window_size)
            return np.convolve(interval, window, 'valid')
        mask_hist = np.squeeze(np.array(self.mopt.mask_history))
        if(ifraw==True):
            mask_hist = np.squeeze(np.array(self.mopt.raw_mask_history))
        new_mask_hist = np.zeros(shape = mask_hist.shape)
        for i in range(self.data_size):
            new_mask_hist[:(len(new_mask_hist)-smoothing_points+1),i] = movingaverage(mask_hist[:,i], smoothing_points)
        plt.figure(figsize=(20,16))
        plt.imshow(new_mask_hist,aspect='auto')
        plt.show()
    def get_most_important_features(self,ifplot=True,return_grad=False,plot_2d_shape =None ):
        m_opt = self.mopt.get_opt_mask(self.unmasked_data_size,self.subnet.model,None,None)  
        m_opt_indices = np.nonzero(m_opt)[0]
        importances = np.zeros((self.data_size))
        if(return_grad == False):
            m_opt_pert_family = np.tile(m_opt,(self.data_size,1))
            m_opt_performance = self.subnet.model.predict(m_opt[None,:])
            for i in range(self.data_size):    
                m_opt_pert_family[i,i] = 1.0 if m_opt[i]==0.0 else 0.0
                importances[i] = self.subnet.model.predict(m_opt_pert_family[i:i+1,:]) 
            if(ifplot==True):
                if plot_2d_shape is None: #if 1D
                    plt.subplot(211)
                    plt.bar(np.arange(self.data_size),importances*m_opt,color=colors)
                    plt.subplot(212)
                    plt.bar(np.arange(self.data_size),importances*(1-m_opt),color=colors)
                    plt.show()
                else:
                    
                    m_opt_indices = np.nonzero(m_opt)[0]
                    vmin = np.min(importances[m_opt_indices])*0.999
                    vmax = np.max(importances[m_opt_indices])                   
                    plt.matshow(np.reshape(importances*m_opt,newshape=plot_2d_shape),vmin=vmin,vmax=vmax,cmap="Reds")
                    plt.colorbar()
                    plt.show()   
                    vmin = np.min(importances[~m_opt_indices])*0.999
                    vmax = np.max(importances[~m_opt_indices])
                    plt.matshow(np.reshape(importances*(1-m_opt)*(-1),newshape=plot_2d_shape),vmin=-vmax,vmax=-vmin,cmap="Reds")
                    plt.colorbar()
                    plt.show()  
                    
                    plt.matshow(np.reshape(importances,newshape=plot_2d_shape))
                    plt.colorbar()
                    plt.show()

        else:
            importances = -MaskOptimizer.gradient(self.subnet.model,m_opt[None,:])[0][0,:]    
            if(ifplot==True):
                if plot_2d_shape is None: #if 1D
                    colors = np.array(["blue"]*self.data_size)
                    colors[np.nonzero(m_opt)[1]] = ["red"]
                    plt.bar(np.arange(self.data_size),importances,color=colors)
                    plt.show()
                else:
                    plt.matshow(np.reshape(importances,newshape=plot_2d_shape))
                    plt.colorbar()
                    plt.show()
                    plt.matshow(np.reshape(m_opt,newshape=plot_2d_shape))
                    plt.show()
        return importances,m_opt