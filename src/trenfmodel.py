import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
import tensorflow_probability as tfp
tfd = tfp.distributions

import sys
sys.path.append('../src/')
import flow_utils as futils
import flows
TransferSpectra = futils.TransferSpectra
SimpleRQSpline = futils.SimpleRQSpline
TRENF = flows.TRENF_classic

###################################################################################
class VItrenf(tf.keras.Model):
    
    def __init__(self, nc, evolve, nlayers=3, name=None, mode='classic', priorv=1., nknots=100, nbins=32, \
                 linknots=False, fitmean=True, fitnoise=True, fitscale=True, meanfield=True, fitoffset=False):
        super(VItrenf, self).__init__(name=name)
        self.nc = nc
        self.prior = tfd.Independent(tfd.MultivariateNormalDiag(loc=tf.zeros((nc, nc, nc, )), \
                                                                scale_diag= priorv* tf.ones((nc, nc, nc, ))), 2)
        if mode == 'classic':
            self.trenf = flows.TRENF_classic(nc, nlayers=nlayers, fitmean=fitmean, fitnoise=fitnoise, nknots=nknots, nbins=nbins, \
                                             fitscale=fitscale, linknots=linknots)
        elif mode == 'affine':
            self.trenf = flows.TRENF_affine(nc, nlayers=nlayers, nknots=nknots, fitmean=fitmean, fitnoise=fitnoise, \
                                            fitscale=fitscale, linknots=linknots, meanfield=meanfield)

        elif mode == 'hybrid':
            self.trenf = flows.TRENF_affine_spline(nc, nlayers=nlayers, nbins=nbins, nknots=nknots, fitmean=fitmean, \
                                                   fitnoise=fitnoise, fitscale=fitscale, linknots=linknots, meanfield=meanfield)
        self.noise = self.trenf.noise
        self.trenf(self.noise.sample(1))
        self.bijector = self.trenf.bijector
        self._variables = self.trenf.variables
        self.evolve = evolve
        #fitoffset

    def transform(self, eps):
        """Transform from noise to parameter space"""
        x = self.bijector.forward(eps)
        return x
    
    def sample(self, n=1):
        """Transform from noise to parameter space"""
        x = self.trenf.sample(n)
        return x
    
    @property    
    def q(self):
        """Variational posterior for the weight"""
        return self.trenf.flow

    @property
    def sample_linear(self):
        z = self.q.sample(1)
        s = self.evolve.z_to_lin(z)
        return s
    
    
    def call(self, z=None, n=1):
        """Predict p(y|x)"""
        if z is None: 
            z = self.q.sample(n)
        s = z_to_lin(z)
        final_field = pm(s)
        std = dnoise
        return z, tfd.Normal(final_field, std)
