"""
[todo] load up vanHatData
[todo] train denoising autoencoder (theano)
[todo] collect denoised ground truth
[todo] shuffle weights and relearn

"""

import numpy as np
import scipy.linalg
import data

import theano
import theano.tensor as T
import settings
from autoencoder import autoencoder

class Mime(object):

    def __init__(self):
        """
        global parameters
        """
        self.T = 10000 # training batch size
        self.myVanHat = data.vanHatData(img_path = settings.data['VH_dir'], base_fname = settings.data['VH_cache'])
        
    def train_ae(self):
        ae = autoencoder(self.myVanHat)
        
if __name__ == '__main__':
    m = Mime()
    m.train_ae()

