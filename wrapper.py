"""
[todo] load up vanHatData
[todo] train denoising autoencoder (theano)
[todo] collect denoised ground truth
[todo] shuffle weights and relearn

"""

from numpy import *
import scipy.linalg
import data

import theano
import theano.tensor as T

class Mime(object):

    def __init__(self):
        """
        global parameters
        """
        self.T = 10000 # training batch size
        
    def get_data(self):
        local_path = '/Users/urs/Dropbox_outsource/vanHarteren/iml00001-04212/'
        myVanHat = data.vanHatData(img_path = local_path, base_fname='/Users/urs/Dropbox_outsource/vanHateren_cache/')
        
        
    def train_ae(self):
        pass
        
        
