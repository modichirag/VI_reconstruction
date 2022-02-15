import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--jobid', type=int, help='an integer for the accumulator')
#parser.add_argument('--rank', type=int, default=0, help='an integer for the accumulator')

args = parser.parse_args()
device = args.jobid
#rank = args.rank

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#print("\nDevice name\n", tf.test.gpu_device_name(), "\n")
print("\nDevices\n", get_available_gpus())


import contextlib
import functools
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import scipy.optimize as sopt


import sys, os
sys.path.append('../src/')
sys.path.append('../../galference/utils/')
sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')

import flowpm
from astropy.cosmology import Planck15
# from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

import tools
import diagnostics as dg
from tfuncs import pm, pmz, z_to_lin

sys.path.append('../../hmc/src/')
from pyhmc import PyHMC
from mala import MALA
from callback import callback_sampling

##########
bs, nc = 200, 32
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
nsims = 200
donbody = False
order = 1
shotnoise = bs**3/nc**3
dnoise = 1 #shotnoise/nc**1.5  
if order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_mala/L%04d_N%04d_LPT/'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_mala/L%04d_N%04d_ZA/'%(bs, nc)
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)


klin = np.loadtxt('../../galference/data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../galference/data//Planck15_a1p00.txt').T[1]

ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                                                           
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh).astype(np.float32)




##############################################
@tf.function
def pm(linear):
    print("PM graph")
    cosmo = flowpm.cosmology.Planck15()
    stages = np.linspace(a0, af, nsteps, endpoint=True)

    if donbody:
        print('Nobdy sim')
        state = lpt_init(cosmo, linear, a=a0, order=2)
        final_state = nbody(cosmo, state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(cosmo, linear, a=af, order=order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return tfinal_field


@tf.function
def z_to_lin(z):
    print(nc)
    whitec = r2c3d(z* nc**1.5)
    lineark = tf.multiply(
        whitec, tf.cast((priorwt / (bs**3))**0.5, whitec.dtype))
    linear = c2r3d(lineark, dtype=tf.float32)
    return linear


@tf.function
def pmz(z):
    print("PM graph")
    linear = z_to_lin(z)
    return pm(linear)


np.random.seed(100)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc)
ic = z_to_lin(zic).numpy()
fin = pm(tf.constant(ic)).numpy()
data = fin + noise
data = data.astype(np.float32)
tfdata = tf.constant(data)
tfnoise = tf.constant(dnoise)
np.save(fpath + 'ic', ic)
np.save(fpath + 'fin', fin)
np.save(fpath + 'data', data)


##############################################
@tf.function
def unnormalized_log_prob(z):
    z = tf.reshape(z, data.shape)
    #Chisq
    final_field = pmz(z)
    residual = (final_field - data)/dnoise
    base = residual
    chisq = tf.multiply(base, base)
    chisq = 0.5 * tf.reduce_sum(chisq)
    logprob = -chisq
    #Prior
    logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(z)) 
    return  logprob + logprior


@tf.function
def grad_log_prob(linear):
    with tf.GradientTape() as tape:
        tape.watch(linear)
        logposterior = unnormalized_log_prob(linear)        
    grad = tape.gradient(logposterior, linear)
    return grad


py_log_prob = lambda x: unnormalized_log_prob(tf.constant(x, dtype=tf.float32)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: grad_log_prob(tf.constant(x, dtype=tf.float32)).numpy().astype(np.float32)

malakernel = MALA(py_log_prob, py_grad_log_prob)


samples, pyacc = [], []
start = time.time()
#print('starting in rank %d of device %d '%(rank, device))

thinning = 50
print("\nstartng MALA in device :%d, rank :"%device, rank)

np.random.seed(100*device+rank + 2021)
x0 = np.random.normal(size=nc**3).reshape(1, nc, nc, nc)
q = x0.copy()
V_q, V_gq = None, None

for i in range(10000):
    stepsize = np.random.uniform(0.01, 0.03, 1)[0]
    q, acc, V_q, V_gq = malakernel.step(q, stepsize, V_q, V_gq)
    #print("Accepted : ", acc)
    if i %thinning == 0: samples.append(q.astype(np.float32))
    pyacc.append(acc)
    if (i%thinning) == 0: 
        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
        #print("Acceptance in device %d, rank %d = "%(device, rank), np.array(pyacc).sum()/len(pyacc))
        print("Acceptance in device %d, rank %d = "%(device, rank), np.unique(pyacc, return_counts=True)[1]/len(pyacc))
        np.save(fpath + '/samples%02d-%02d'%(device, rank), np.array(samples))
        fig = callback_sampling([z_to_lin(i) for i in samples[-10:]], ic, bs)
        plt.savefig(fpath + '/figs/iter%02d-%05d'%(device, i))
        plt.close()
samples = np.array(samples)
np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
end = time.time()
print('Time taken in rank %d= '%rank, end-start)
