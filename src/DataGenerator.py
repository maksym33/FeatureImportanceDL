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
class UniformDataGenerator():
    def __init__(self,x_batch_size,mask_batch_size,x_shape,y_generator_function,unmasked_x_size,x_range=[-1,1]):        
        self.x_batch_size = x_batch_size
        self.mask_batch_size = mask_batch_size
        self.x_shape = x_shape
        self.y_generator = y_generator_function
        self.unmasked_x_size = unmasked_x_size
        self.x_range = x_range
        self.x_size = np.zeros(shape=self.x_shape).size
        self.batch_shape = (self.x_batch_size,) + self.x_shape
        print("Data generator x batch shape:\t "+str(self.batch_shape))
        print("Data generator mask batch shape: "+str((self.mask_batch_size,) + self.x_shape))
        print("Data generator y batch shape:\t "+str((self.x_batch_size,) + ("?",)))
    #def get_masks(self):
    #    masks_zero = np.zeros(shape = (self.mask_batch_size,self.x_size-self.unmasked_x_size))
    #    masks_one = np.ones(shape = (self.mask_batch_size,self.unmasked_x_size))
    #    masks = np.concatenate([masks_zero,masks_one],axis = 1)
    #    masks_permuted = np.apply_along_axis(np.random.permutation,1,masks)
    #    return masks_permuted
    def get_x(self,how_many=0):
        if(how_many==0):
            how_many = self.x_batch_size
        x_raw = np.random.uniform(self.x_range[0],self.x_range[1],how_many*self.x_size)
        return np.reshape(x_raw,newshape=(how_many,self.x_size))
    def get_y(self,x,how_many=0):
        if(how_many==0):
            how_many = self.x_batch_size        
        x = np.reshape(x,newshape=(how_many,self.x_size))
        return np.apply_along_axis(self.y_generator,1,x)
    def get_batch(self,how_many=0):
        x=self.get_x(how_many)
        y=self.get_y(x,how_many)
        #if (self.ifgenX==True): # if y modifies x then gen_y return a list [y,new_x]
        #    new_y = y[:,0]
        #    x = y[:,1]
        #    y=new_y
        return x,y
