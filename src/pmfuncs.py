import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions


import sys, os
#sys.path.append('../../utils/')
sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-rk4')

import flowpm
from astropy.cosmology import Planck15
# from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
from flowpm.tfpower import linear_matter_power
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
#import tools


class Evolve():
    
    def __init__(self, nc, bs, a0=0.1, af=1., nsteps=5, donbody=False, order=2, cosmodict=None, dtype=np.float32):
        self.nc, self.bs = nc, bs
        #self.ipklin = ipklin
        self.a0, self.af, self.nsteps = a0, af, nsteps
        self.stages = np.linspace(a0, af, nsteps, endpoint=True)
        self.donbody = donbody
        if cosmodict is None: self.cosmodict = flowpm.cosmology.Planck15().to_dict()
        else: self.cosmodict = cosmodict
        self.order = order
        self.dtype = dtype
        self._build()

    def _build(self):
        nc, bs = self.nc, self.bs
        self.cosmo = flowpm.cosmology.Planck15(**self.cosmodict)        
        self.klin = np.logspace(-5, 3, 2000)
        self.plin = linear_matter_power(self.cosmo, self.klin)
        self.ipklin = iuspline(self.klin, self.plin)
        #kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
        kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
        self.kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5).astype(self.dtype)
        self.pkmesh = self.ipklin(self.kmesh).astype(self.dtype)
        self.pkmesh_r = self.ipklin(self.kmesh)[:, :, :nc//2+1].astype(self.dtype)


    @tf.function
    def pm(self, linear, cosmodict=None):
        print("PM graph")
        if cosmodict is None:
            cosmo = flowpm.cosmology.Planck15(**self.cosmodict)
        else:
            cosmo = flowpm.cosmology.Planck15(**cosmodict)

        if self.donbody:
            print('Nobdy sim')
            state = lpt_init(cosmo, linear, a=self.a0, order=self.order)
            final_state = nbody(cosmo, state,  self.stages, self.nc)
        else:
            print('ZA/2LPT sim')
            final_state = lpt_init(cosmo, linear, a=self.af, order=self.order)
        tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
        return tfinal_field



    @tf.function
    def z_to_lin(self, z):
        whitec = r2c3d(z* self.nc**1.5)
        lineark = tf.multiply(
            whitec, tf.cast((self.pkmesh / (self.bs**3))**0.5, whitec.dtype))
        linear = c2r3d(lineark)
        return linear


    @tf.function
    def pmz(self, z):
        print("PM graph")
        linear = self.z_to_lin(z)
        return self.pm(linear)

    @tf.function
    def pmzk(self, zk):
        print("PM graph")
        linear = self.zk_to_lin(zk)
        return self.pm(linear)


    @tf.function
    def zk_to_z(self, zk):
        #print("zk to linl : ", zk.shape, self.pkmesh_r.shape)
        zkn = zk * (self.nc**3/2)**0.5
        zknc = tf.complex(zkn[..., 0], zkn[..., 1])
        z = tf.signal.irfft3d(zknc)
        return z

    @tf.function
    def z_to_zk(self, z):
        zknc = tf.signal.rfft3d(z)
        zkn0, zkn1 = tf.math.real(zknc), tf.math.imag(zknc)
        zkn = tf.stack([zkn0, zkn1], -1)    
        zk = zkn / (self.nc**3/2.)**0.5
        return zk

    @tf.function
    def zk_to_link(self, zk):
        #print("zk to linl : ", zk.shape, self.pkmesh_r.shape)
        zkn = zk * (self.nc**3/2)**0.5
        link = zkn * (tf.expand_dims(self.pkmesh_r, -1)/self.bs**3)**0.5  * self.nc**1.5
        return link

    @tf.function
    def zk_to_lin(self, zk):
        #print("zk to lin : ", zk.shape)
        link = self.zk_to_link(zk)
        linkc = tf.complex(link[..., 0], link[..., 1])
        lin = tf.signal.irfft3d(linkc)
        return lin


    @tf.function
    def link_to_zk(self, link):
        link = link / self.nc**1.5
        zkn = link / (tf.expand_dims(tf.cast(self.pkmesh_r, link.dtype), -1) /self.bs**3)**0.5 
        zk = zkn / (self.nc**3/2)**0.5
        return zk

    @tf.function
    def lin_to_zk(self, lin):
        linkc = tf.signal.rfft3d(lin) 
        link0, link1 = tf.math.real(linkc), tf.math.imag(linkc)
        link = tf.stack([link0, link1], -1)
        return self.link_to_zk(link)


    @tf.function
    def zdist(self):
        return tfd.Normal(0, 1)
    

    #@tf.function
    #def zkdist(self):
        #self.scalezk = nc**1.5/2**0.5
        #return tf.Normal(0, self.scalezk)




class Evolve_bias():
    
    def __init__(self, nc, bs, a0=0.1, af=1., nsteps=5, donbody=False, order=2, cosmodict=None, dtype=np.float32):
        self.nc, self.bs = nc, bs
        #self.ipklin = ipklin
        self.a0, self.af, self.nsteps = a0, af, nsteps
        self.stages = np.linspace(a0, af, nsteps, endpoint=True)
        self.donbody = donbody
        if cosmodict is None: self.cosmodict = flowpm.cosmology.Planck15().to_dict()
        else: self.cosmodict = cosmodict
        self.order = order
        self.dtype = dtype
        self._build()

    def _build(self):
        nc, bs = self.nc, self.bs
        self.cosmo = flowpm.cosmology.Planck15(**self.cosmodict)        
        self.klin = np.logspace(-5, 3, 2000)
        self.plin = linear_matter_power(self.cosmo, self.klin)
        self.ipklin = iuspline(self.klin, self.plin)
        #kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
        kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
        self.kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5).astype(self.dtype)
        self.pkmesh = self.ipklin(self.kmesh).astype(self.dtype)
        self.pkmesh_r = self.ipklin(self.kmesh)[:, :, :nc//2+1].astype(self.dtype)


    @tf.function
    def pm(self, linear, cosmodict=None, returnpos=False):
        print("PM graph")
        if cosmodict is None:
            cosmo = flowpm.cosmology.Planck15(**self.cosmodict)
        else:
            cosmo = flowpm.cosmology.Planck15(**cosmodict)

        if self.donbody:
            print('Nobdy sim')
            state = lpt_init(cosmo, linear, a=self.a0, order=self.order)
            final_state = nbody(cosmo, state,  self.stages, self.nc)
        else:
            print('ZA/2LPT sim')
            final_state = lpt_init(cosmo, linear, a=self.af, order=self.order)
        tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
        if returnpos: return tfinal_field, final_state[0]
        return tfinal_field


    @tf.function
    def biasfield(self, linear, bias, cosmodict=None):
        print("PM graph")
        if cosmodict is None:
            cosmo = flowpm.cosmology.Planck15(**self.cosmodict)
        else:
            cosmo = flowpm.cosmology.Planck15(**cosmodict)

        if self.donbody:
            print('Nobdy sim')
            state = lpt_init(cosmo, linear, a=self.a0, order=self.order)
            final_state = nbody(cosmo, state,  self.stages, self.nc)
        else:
            print('ZA/2LPT sim')
            final_state = lpt_init(cosmo, linear, a=self.af, order=self.order)
    
        b1, b2 = bias[0], bias[1]
        fpos = final_state[0]
        w0 =  tf.reshape(linear, (linear.shape[0], -1))
        w0 = w0 - tf.expand_dims(tf.reduce_mean(w0, 1), -1)
        w2  = w0*w0
        w2 = w2 - tf.expand_dims(tf.reduce_mean(w2, 1), -1)
        weight = b1*w0 + b2*w2
        bmodel = cic_paint(tf.zeros_like(linear), fpos, weight = weight)
        return bmodel




    @tf.function
    def z_to_lin(self, z):
        whitec = r2c3d(z* self.nc**1.5)
        lineark = tf.multiply(
            whitec, tf.cast((self.pkmesh / (self.bs**3))**0.5, whitec.dtype))
        linear = c2r3d(lineark)
        return linear


    @tf.function
    def lin_to_z(self, linear):
        lineark= r2c3d(linear)
        whitec  = tf.multiply(
            lineark, 1/tf.cast((self.pkmesh / (self.bs**3))**0.5, lineark.dtype))
        z = c2r3d(whitec)
        z = z/ self.nc**1.5
        return z


    @tf.function
    def pmz(self, z):
        print("PM graph")
        linear = self.z_to_lin(z)
        return self.pm(linear)

    @tf.function
    def pmzk(self, zk):
        print("PM graph")
        linear = self.zk_to_lin(zk)
        return self.pm(linear)


    @tf.function
    def zk_to_z(self, zk):
        #print("zk to linl : ", zk.shape, self.pkmesh_r.shape)
        zkn = zk * (self.nc**3/2)**0.5
        zknc = tf.complex(zkn[..., 0], zkn[..., 1])
        z = tf.signal.irfft3d(zknc)
        return z

    @tf.function
    def z_to_zk(self, z):
        zknc = tf.signal.rfft3d(z)
        zkn0, zkn1 = tf.math.real(zknc), tf.math.imag(zknc)
        zkn = tf.stack([zkn0, zkn1], -1)    
        zk = zkn / (self.nc**3/2.)**0.5
        return zk

    @tf.function
    def zk_to_link(self, zk):
        #print("zk to linl : ", zk.shape, self.pkmesh_r.shape)
        zkn = zk * (self.nc**3/2)**0.5
        link = zkn * (tf.expand_dims(self.pkmesh_r, -1)/self.bs**3)**0.5  * self.nc**1.5
        return link

    @tf.function
    def zk_to_lin(self, zk):
        #print("zk to lin : ", zk.shape)
        link = self.zk_to_link(zk)
        linkc = tf.complex(link[..., 0], link[..., 1])
        lin = tf.signal.irfft3d(linkc)
        return lin


    @tf.function
    def link_to_zk(self, link):
        link = link / self.nc**1.5
        zkn = link / (tf.expand_dims(tf.cast(self.pkmesh_r, link.dtype), -1) /self.bs**3)**0.5 
        zk = zkn / (self.nc**3/2)**0.5
        return zk

    @tf.function
    def lin_to_zk(self, lin):
        linkc = tf.signal.rfft3d(lin) 
        link0, link1 = tf.math.real(linkc), tf.math.imag(linkc)
        link = tf.stack([link0, link1], -1)
        return self.link_to_zk(link)


    @tf.function
    def zdist(self):
        return tfd.Normal(0, 1)
    
