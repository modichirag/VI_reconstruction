import numpy as np
import matplotlib

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import tensorflow_probability as tfp
tfd = tfp.distributions



import sys, os
sys.path.append('../../galference/utils/')
sys.path.append('../src/')
sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')

import flowpm
from astropy.cosmology import Planck15
# from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

from pmfuncs import Evolve

import tools
import diagnostics as dg
from pmfuncs import Evolve

import contextlib
import functools
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import scipy.optimize as sopt


@tf.function
def recon_prototype_z(z, data, Rsm, evolve, dnoise=1.):
    """Return the loss given white noise and data 
    """
    bs, nc = evolve.bs, evolve.nc
    kmesh = evolve.kmesh
    #z = tf.reshape(z, data.shape)
    final_field = evolve.pmz(z)

    residual = (final_field - data)/dnoise
    base = residual
 
    if True:
        print("Add annealing section to graph")
        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
        smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
        basek = r2c3d(base, norm=nc**3)
        basek = tf.multiply(basek, tf.cast(smwts, basek.dtype))
        base = c2r3d(basek, norm=nc**3)

    chisq = tf.multiply(base, base)
    chisq = 0.5 * tf.reduce_sum(chisq, axis=(-3, -2, -1))
    logprob = -chisq
    #Prior
    logprior = tf.reduce_sum(tfd.Normal(tf.cast(0., z.dtype), 1.).log_prob(z), axis=(-3, -2, -1)) 


    loss = -1.* (logprob + logprior)
    return loss, chisq, logprior

@tf.function
def val_and_grad_z(x, data, Rsm, evolve):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_prototype_z(x, data, Rsm, evolve)[0]
    grad = tape.gradient(loss, x)
    return loss, grad



###########Minimize
def map_estimate(evolve, x0, data, RR, method='adam', lr=0.01, maxiter=200):

    if x0.dtype == np.float32: dtype = tf.float32
    elif x0.dtype == np.float64: dtype = tf.float64
    print("Mininize for RR = ", RR)
    if method == 'lbfgs':
        print("Using LBFGS")
        def min_lbfgs_sicpy(x0):
            return [vv.numpy().astype(np.float64)  for \
                    vv in val_and_grad_z(tf.constant(x0, dtype=dtype), 
                                         tf.constant(data, dtype=dtype), 
                                         tf.constant(RR, dtype=dtype), evolve)]                                          

        ##THIS MIGHT BARF SINCE recon_prototype_z does not reshape the vector correctly. Its commented.
        results = sopt.minimize(fun=min_lbfgs_sicpy, x0=x0, jac=True, method='L-BFGS-B',
                                    tol=1e-10,                                                                                                               
                                options={'maxiter':maxiter, 'ftol': 1e-12, 'gtol': 1e-12, 'eps':1e-12})

        linearz = results.x.reshape(data.shape)
        x0 = linearz*1.

    else:
        print("Using ADAM")
        nc = evolve.nc
        linearz = tf.Variable(name='linmesh', shape=x0.shape, dtype=dtype,
                              initial_value=x0, trainable=True)
        
        losses = []
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        for i in range(maxiter):
            loss, grads = val_and_grad_z(linearz, 
                                       tf.constant(data, dtype=dtype), 
                                       tf.constant(RR, dtype=dtype), evolve)
            opt.apply_gradients(zip([grads], [linearz]))
            losses.append(loss)
        x0 = linearz.numpy()

    return x0 



#############################
def map_estimate_halo(x0, RR, val_and_grad, method='adam', lr=0.01, maxiter=200):

    if x0.dtype == np.float32: dtype = tf.float32
    elif x0.dtype == np.float64: dtype = tf.float64
    print("Mininize for RR = ", RR)
    print("Using ADAM")
    linearz = tf.Variable(name='linmesh', shape=x0.shape, dtype=dtype,
                          initial_value=x0, trainable=True)
        
    print("linearz shape : ", linearz.shape)
    losses = []
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    for i in range(maxiter):
        logprob, grads = val_and_grad(linearz, tf.constant(RR, dtype=dtype))
        grads = grads*-1.
        loss  = logprob*-1.
        opt.apply_gradients(zip([grads], [linearz]))
        losses.append(loss)
    x0 = linearz.numpy()
    return x0 
