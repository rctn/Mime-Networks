"""
[todo] load up vanHatData
[todo] train denoising autoencoder (theano)
[todo] collect denoised ground truth
[todo] shuffle weights and relearn

"""

import numpy as np
import data
import settings
from autoencoder import autoencoder
import sfo

class Mime(object):

    def __init__(self):
        """
        global parameters
        """
        self.T = 10000 # training batch size
        self.myVanHat = data.vanHatData(img_path = settings.data['VH_dir'], base_fname = settings.data['VH_cache'],)
        
    def train_ae(self):
        ae = autoencoder(self.myVanHat)
        return ae
        
if __name__ == '__main__':
    m = Mime()
    m.train_ae()
    ae = m.train_ae()
    f, df = ae.f_df(ae.theta, (0,10))
    batches = [(i, (i+1)*100) for i in range(100)]

    sfopt = sfo.SFO(ae.f_df, ae.theta, batches)
    theta = sfopt.optimize()

