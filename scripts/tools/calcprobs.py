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



import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--suffix', type=str, help='suffix')
parser.add_argument('--jobid', type=int, default=1, help='an integer for the accumulator')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')

args = parser.parse_args()
device = args.jobid
#suffix = args.suffix

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#print("\nDevice name\n", tf.test.gpu_device_name(), "\n")
print("\nDevices\n", get_available_gpus())


from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import scipy.optimize as sopt

import sys, os, flowpm
sys.path.append('../../galference/utils/')
import tools
import diagnostics as dg

#sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')
#import flowpm

sys.path.append('../src/')
import trenfmodel
from pmfuncs import Evolve
from hmcfuncs import DM_config, DM_fourier, Kwts
from pyhmc import PyHMC, PyHMC_batch
from callback import callback, datafig, callback_fvi, callback_sampling

##########
bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
donbody = False
order = 1
shotnoise = bs**3/nc**3
dnoise = 1. #shotnoise/nc**1.5  


klin = np.loadtxt('../../galference/data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../galference/data//Planck15_a1p00.txt').T[1]

ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                                                           
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh).astype(np.float32)
evolve = Evolve(nc, bs,  a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     


##############################################

print("\nFor seed : ", args.seed)
np.random.seed(args.seed)

#samplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-kwts4-fourier/'
#samplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-kwts4-fourier-corr/'
samplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA-%s/'%(args.bs, args.nc, args.suffix)
dnoise = 1.
ic = np.load(samplespath + '/ic.npy')
fin = np.load(samplespath + '/fin.npy')
data = np.load(samplespath + '/data.npy')
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
tfdata = tf.constant(data)

dmfuncs = DM_fourier(evolve, tfdata, dnoise=dnoise)
#kwts = Kwts(evolve, mode=args.kwts, knoise=knoise)
py_log_prob = lambda x: dmfuncs.unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: dmfuncs.grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
#hmckernel = PyHMC_batch(py_log_prob, py_grad_log_prob, invmetric_diag=kwts.kwts, returnV=True)




###################################################################################
    
#Train FLOW using previos samples assuming we have them
for i in range(4):
    #truesamples.append(np.load('/mnt/ceph/users/cmodi/galference/dm_hmc/L1000_N0128_ZA-kwts4-fourier/samples%d-00.npy'%i)[50:, 0])
    samples = np.load(samplespath + '/samples%d-00.npy'%i)
    print(samples.shape)
    lps = []
    for j in range(samples.shape[0]):
        lps.append(py_log_prob(samples[j]))
        print(lps[-1])
    np.save(samplespath + '/lps%d-00'%i, np.array(lps))
#$#samples = np.stack(samples, axis=1).astype(np.float32)[50:]
#$#print("True samples shape : ", samples.shape)
#$#
#$#lps = []
#$#
#$#for j in range(samples.shape[0]):
#$#    lps.append(dmfuncs.unnormalized_log_prob(samples[i]))
#$#    print(lps[-1])
#$#               
