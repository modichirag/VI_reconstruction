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
parser.add_argument('--jobid', type=int, default=1, help='an integer for the accumulator')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--repvi', type=int, default=0, help="reparametrization gradient")

args = parser.parse_args()
device = args.jobid
suffix = args.suffix

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
sys.path.append('../../galference/utils/')
import tools
import diagnostics as dg

sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')

import flowpm
from astropy.cosmology import Planck15
# from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

sys.path.append('../src/')
import flow_utils as futils
import flows
TransferSpectra = futils.TransferSpectra
SimpleRQSpline = futils.SimpleRQSpline
#TRENF = flows.TRENF_classic
from tfuncs import pm, pmz, z_to_lin

sys.path.append('../../hmc/src/')
from pyhmc import PyHMC

from callback import callback, datafig

##########
bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
nsims = 200
donbody = False
order = 1
shotnoise = bs**3/nc**3
dnoise = 1. #shotnoise/nc**1.5  
if order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_vi/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_vi/L%04d_N%04d_ZA'%(bs, nc)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)


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


print("\nFor seed : ", args.seed)
np.random.seed(args.seed)
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
fig = datafig(ic, fin, data, bs, dnoise)
plt.savefig(fpath + 'data')
plt.close()

###################################################################################
class VItrenf(tf.keras.Model):
    
    def __init__(self, nc, nlayers=3, name=None):
        super(VItrenf, self).__init__(name=name)
        self.nc = nc
        self.prior = tfd.Independent(tfd.MultivariateNormalDiag(loc=tf.zeros((nc, nc, nc, )), \
                                                                scale_diag=tf.ones((nc, nc, nc, ))), 2)
        self.trenf = flows.TRENF_classic(nc, nlayers=nlayers, fitmean=True, fitnoise=True)
        #self.trenf = flows.TRENF_affine(nc, nlayers=nlayers, fitmean=True, fitnoise=True)
        self.noise = self.trenf.noise
        self.trenf(self.noise.sample(1))
        self.bijector = self.trenf.bijector
        self._variables = self.trenf.variables


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
        s = z_to_lin(z)
        return s
    
    
    def call(self, z=None, n=1):
        """Predict p(y|x)"""
        if z is None: 
            z = self.q.sample(n)
        s = z_to_lin(z)
        final_field = pm(s)
        std = dnoise
        return z, tfd.Normal(final_field, std)
    


##@tf.function
##def likelihood(data, z, Rsm):
##    final_field = pmz(z)
###    if Rsm != 0:
###        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
###        smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
###        basek = r2c3d(final_field, norm=nc**3)
###        basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
###        final_field = c2r3d(basek, norm=nc**3)
##    logprob = tfd.Normal(final_field, scale=dnoise).log_prob(data)
##    return tf.reduce_sum(logprob, axis=[-3, -2, -1])
##
@tf.function
def likelihood(data, z, noiselevel):
    final_field = pmz(z)
#    if Rsm != 0:
#        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
#        smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
#        basek = r2c3d(final_field, norm=nc**3)
#        basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
#        final_field = c2r3d(basek, norm=nc**3)
    logprob = tfd.Normal(final_field, scale=dnoise*tf.math.sqrt(noiselevel)).log_prob(data)
    return tf.reduce_sum(logprob, axis=[-3, -2, -1])


    
@tf.function
def bbvi_trenf(model, likelihood, data, Rsm, nsamples = tf.constant(1)):   
    print("\nUse full gradeint of ELBO\n")    
    elbos = []
    nsamples = 1
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        sample = model.q.sample(nsamples)*1.
        logl = tf.reduce_mean(likelihood(data, sample, Rsm))
        logp = tf.reduce_mean(model.prior.log_prob(sample))
        logq = tf.reduce_mean(model.q.log_prob(sample))
        elbo = logl + logp - logq
        print(logl, logp, logq, elbo)
        elbo = -1 * elbo
        elbos.append(elbo)
    grad = tape.gradient(elbo, model.trainable_variables)
    return tf.stack([logl, logp, logq, -elbo]), grad


@tf.function
def bbvi_trenf_rp(model, likelihood, data, Rsm, nsamples = tf.constant(1)):   
    print("\nUse reparametrized gradient\n")
    elbos = []
    nsamples = 1
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        sample = model.q.sample(nsamples)*1.
        logl = tf.reduce_mean(likelihood(data, sample, Rsm))
        logp = tf.reduce_mean(model.prior.log_prob(sample))
        logq = tf.reduce_mean(model.q.log_prob(sample))
        elbo = logl + logp - logq
        print(logl, logp, logq, elbo)
        elbo = -1 * elbo
        elbos.append(elbo)
    gradz = tape.gradient(elbo, sample)
    grad = tape.gradient(sample, model.trainable_variables, gradz)
    return tf.stack([logl, logp, logq, -elbo]), grad



def optimizetrenf(model, data, opt,  Rsm, niter=100, callback=callback, citer=10, nsamples=1, saveiter=1, repvi=args.repvi):
    
    losses = []
    spath = fpath + 'R%02d/'%(Rsm*10)
    os.makedirs(spath, exist_ok=True)
    os.makedirs(spath + '/opt', exist_ok=True)
    print('\nDoing for smoothing of R=%d\n'%Rsm)
    start = time.time()
    for epoch in range(niter+1):
        #print(epoch)
        if epoch%100 == 0: print(epoch)
        if repvi : loss, grad = bbvi_trenf_rp(model, likelihood, data, Rsm, nsamples=tf.constant(nsamples))
        else: loss, grad = bbvi_trenf(model, likelihood, data, Rsm, nsamples=tf.constant(nsamples))
        #for g,v in zip(grad, model.trainable_variables):
        #    tf.print(v.name, tf.reduce_max(g))
        opt.apply_gradients(zip(grad, model.trainable_variables))
        losses.append(loss/nc**3)
        if epoch%citer == 0: 
            print("LogL, Logp, Logq, ELBO at epoch %d is: "%epoch, loss.numpy())
            print("Time taken for %d iterations : "%citer, time.time() - start)
            fig = callback(model, ic, bs, losses)
            plt.savefig(spath + 'iter%05d'%epoch)
            plt.close()
            np.save(spath + 'loss', np.array(losses))
            start = time.time()
        if epoch%saveiter == 0: 
            model.save_weights(spath + '/weights/iter%04d'%(epoch//saveiter))
            np.save(spath + '/opt/iter%04d'%(epoch//saveiter), opt.get_weights(), allow_pickle=True)
            #print('Weights saved')
            
    return losses 




###################################################################################
#######VI
def gauss_smooth(x, Rsm):
    Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
    smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
    basek = r2c3d(x, norm=nc**3)
    basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
    x = c2r3d(basek, norm=nc**3)
    return x


vitrenf = VItrenf(nc, nlayers=3)
for i in vitrenf.trainable_variables:
    print(i.name, i.shape)

plt.imshow(vitrenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig(fpath + 'initsample')
plt.colorbar()
plt.close()

print("\nStart VI\n")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-3,
    decay_steps=200,
    decay_rate=0.96,
    staircase=True)

#opt = tf.keras.optimizers.Adam(learning_rate= 1e-2)

#opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
allRs = [0., 1., 2., 4.]
allRs = [1., 10., 100., 1000.,]
RR = allRs[:args.nR + 1][::-1]

for R in RR:
    print('\nFor smoothing with kernel R = %d'%R)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    #dataR = gauss_smooth(data, R)
    losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 5001, callback, citer=100)

#R = 0.
#losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 1001, callback, citer=10)
    
#opt = tf.keras.optimizers.Adam(learning_rate=0.01)
#losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(0.), 10001, callback, citer=100)

