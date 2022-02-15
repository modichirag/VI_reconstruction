###Use the samples generated from HMC to train VI model.
###This is useful to study if the network is flexible enough to
###learn the distribution of interest

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
parser.add_argument('--jobid', type=int, default=1, help='an integer for the accumulator')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--nlayers', type=int, default=3, help="number of trenf layers")
parser.add_argument('--nbins', type=int, default=32, help="number of bins in trenf spline")
parser.add_argument('--mode', type=str, default="classic", help='')
parser.add_argument('--nknots', type=int, default=100, help='number of trenf layers')
parser.add_argument('--linknots', type=int, default=0, help='linear spacing for knots')
parser.add_argument('--kwts', type=int, default=4, help='number of trenf layers')
parser.add_argument('--fitnoise', type=int, default=1, help='fitnoise')
parser.add_argument('--fitscale', type=int, default=1, help='fitscale')
parser.add_argument('--fitmean', type=int, default=1, help='fitmean')
parser.add_argument('--meanfield', type=int, default=1, help='meanfield for affine')
parser.add_argument('--ntrain', type=int, default=2000, help='number of training iterations')
parser.add_argument('--regwt0', type=float, default=0., help='regularization weight')

args = parser.parse_args()
device = args.jobid
suffix = args.suffix

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
if order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_fvi_samples/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_fvi_samples/L%04d_N%04d_ZA'%(bs, nc)
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
evolve = Evolve(nc, bs,  a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     


##############################################

print("\nFor seed : ", args.seed)
np.random.seed(args.seed)

truesamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA-kwts4-fourier/'%(bs, nc)
trainsamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA-kwts4-fourier-corr/'%(bs, nc)
#truesamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-kwts4-fourier/'
#trainsamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-kwts4-fourier-corr/'
#Invert
#trainsamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-kwts4-fourier/'
#truesamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-kwts4-fourier-corr/'
dnoise = 1.
ic = np.load(truesamplespath + '/ic.npy')
fin = np.load(truesamplespath + '/fin.npy')
data = np.load(truesamplespath + '/data.npy')
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
tfdata = tf.constant(data)
k, pic = tools.power(ic[0], boxsize=bs)
k, pf = tools.power(fin[0], boxsize=bs)
k, pd = tools.power(data[0], boxsize=bs)
k, pn = tools.power(1+noise[0], boxsize=bs)
knoise = evolve.kmesh[evolve.kmesh > k[(pn > pf)][0]].min()
print("Noise dominated after : ", knoise, (evolve.kmesh > knoise).sum()/nc**3)

dmfuncs = DM_fourier(evolve, tfdata, dnoise=dnoise)
kwts = Kwts(evolve, mode=args.kwts, knoise=knoise)
py_log_prob = lambda x: dmfuncs.unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: dmfuncs.grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
hmckernel = PyHMC_batch(py_log_prob, py_grad_log_prob, invmetric_diag=kwts.kwts, returnV=True)


@tf.function
def fvi_train(model, samples, opt):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logq = - tf.reduce_mean(model.q.log_prob(samples))
    gradients = tape.gradient(logq, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return logq

@tf.function
def fvi_train_reg(model, samples, opt, regwt, samples2):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        logq = - tf.reduce_mean(model.q.log_prob(samples))
        logq2 = - tf.reduce_mean(model.q.log_prob(samples2))
        samplevi = model.sample(samples.shape[0])*1.
        logqvi = - tf.reduce_mean(model.q.log_prob(samplevi))
        reg = regwt * tf.reduce_mean((logq2 - logqvi)**2.)
        loss = logq + reg
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return logq


@tf.function
def fvi_train_l2(model, offset, samples, logp, opt):

    variables = model.trainable_variables + [offset]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(variables)
        #logq = - tf.reduce_mean(model.q.log_prob(samples))
        logq = model.q.log_prob(samples)
        loss = tf.reduce_sum((logp - logq - nc**3*offset)**2)
    gradients = tape.gradient(loss, variables)
    opt.apply_gradients(zip(gradients, variables))
    return loss



###################################################################################

vitrenf = trenfmodel.VItrenf(nc, nlayers=args.nlayers, evolve=evolve, nbins=args.nbins, nknots=args.nknots, mode=args.mode, \
                             linknots=bool(args.linknots), fitnoise=bool(args.fitnoise), fitscale=bool(args.fitscale), \
                             fitmean=bool(args.fitmean), meanfield=bool(args.meanfield))
offset = tf.Variable(0.)*0.

for i in vitrenf.variables:
    print(i.name, i.shape)
plt.imshow(vitrenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig(fpath + 'initsample')
plt.colorbar()
plt.close()

print("\nStart VI\n")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-2,
    decay_steps=100,
    decay_rate=0.8,
    staircase=True)


    
#Train FLOW using previos samples assuming we have them
truesamples, truesamplesz = [], []
truelps = []
nchains = 4
for i in range(nchains):
    truesamples.append(np.load(truesamplespath + '/samples%d-00.npy'%i)[:, 0])
    #truelps.append(np.load(truesamplespath + '/lps%d-00.npy'%i)[:, 0])
    try: truesamplesz.append(np.load(truesamplespath + '/samplesz%d-00.npy'%i)[:])
    except:
        z = np.array([evolve.zk_to_z(zzk) for zzk in truesamples[-1]])
        np.save(truesamplespath + '/samplesz%d-00'%i, z)
        truesamplesz.append(z)
truesamples = np.stack(truesamples, axis=1).astype(np.float32)[50:]
truesamplesz = np.stack(truesamplesz, axis=1).astype(np.float32)[50:]
#truelps = np.stack(truelps, axis=1).astype(np.float32)[50:]
#truesamplesz = np.array([evolve.zk_to_z(zzk) for zzk in truesamples])
print("True samples shape : ", truesamples.shape)
print("True samplesz shape : ", truesamplesz.shape)
#print("True lps shape : ", truelps.shape)

trainsamples = []
trainsamplesz = []
trainlps = []
for i in range(nchains):
    trainsamples.append(np.load(trainsamplespath + '/samples%d-00.npy'%i)[:, 0])
    #trainlps.append(np.load(trainsamplespath + '/lps%d-00.npy'%i)[:, 0])
    try: trainsamplesz.append(np.load(trainsamplespath + '/samplesz%d-00.npy'%i))
    except:
        z = np.array([evolve.zk_to_z(zzk) for zzk in trainsamples[-1]])
        np.save(trainsamplespath + '/samplesz%d-00'%i, z)
        trainsamplesz.append(z)
    
trainsamples = np.stack(trainsamples, axis=1).astype(np.float32)[50:]
trainsamplesz = np.stack(trainsamplesz, axis=1).astype(np.float32)[50:]
#trainlps = np.stack(trainlps, axis=1).astype(np.float32)[50:]
print("Train samples shape : ", trainsamples.shape)
print("Train samplesz shape : ", trainsamplesz.shape)
#print("Train lps shape : ", trainlps.shape)
if nc == 128: nsamples = 1
else: nsamples = 4
ntrain = args.ntrain
losses = []

opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)
saveiter = 1000

for i in range(ntrain):
    idx = np.random.choice(trainsamplesz.shape[0], nsamples)
    #batchlps = tf.concat(trainlps[idx], axis=0)
    batch = tf.concat(trainsamplesz[idx], axis=0)
    if args.regwt0 == 0: 
        losses.append(fvi_train(vitrenf, batch, opt))
    else: 
        idx2 = np.random.choice(trainsamplesz.shape[0], nsamples)
        batch2 = tf.concat(trainsamplesz[idx2], axis=0)
        regwt = tf.constant(1*args.regwt0*i/ntrain)
        #if i < ntrain //2 : regwt = regwt * 0.
        losses.append(fvi_train_reg(vitrenf, batch, opt, regwt, batch2))


    if i%100 == 0: 
        print("Iteration %d"%i)
        fig = callback_fvi(vitrenf, ic, bs, losses)
        plt.savefig(fpath + 'trainr%05d'%i)
        plt.close()

        #batch = tf.concat(trainsamplesz[np.random.choice(trainsamplesz.shape[0], nsamples)], axis=0)
        lqs = []
        for j in range(50):
            batch = tf.concat(trainsamplesz[j*(trainsamplesz.shape[0]//50)], axis=0)
            if j==0: print(batch.shape)
            lq = vitrenf.q.log_prob(batch).numpy() #+ offset.numpy()*nc**3
            if j==0: print(lq.shape)
            lqs.append(lq)
        plt.plot(np.array(lqs).flatten(), ".", label='train samples', alpha=0.5)
        
        lqs = []
        #batch = tf.concat(truesamplesz[np.random.choice(truesamplesz.shape[0], nsamples)], axis=0)
        #lq = vitrenf.q.log_prob(batch).numpy() + offset.numpy()*nc**3
        for j in range(50):
            batch = tf.concat(truesamplesz[j*(truesamplesz.shape[0]//50)], axis=0)
            if j==0: print(batch.shape)
            lq = vitrenf.q.log_prob(batch).numpy() #+ offset.numpy()*nc**3
            if j==0: print(lq.shape)
            lqs.append(lq)
        plt.plot(np.array(lqs).flatten(), ".", label='true samples', alpha=0.5)
        
        lqs = []
        for j in range(50):
            batch = vitrenf.sample(nsamples)
            if j==0: print(batch.shape)
            lq = vitrenf.q.log_prob(batch).numpy() #+ offset.numpy()*nc**3
            if j==0: print(lq.shape)
            lqs.append(lq)
        plt.plot(np.array(lqs).flatten(), '.', label='flow samples', alpha=0.5)

        plt.legend()
        plt.grid(which='both')
        plt.savefig(fpath + 'logq%05d'%i)
        plt.close()
        #$print(batchlps, py_log_prob(evolve.z_to_zk(batch[:nchains])))

    if (i > 0) & (i%saveiter == 0): 
        vitrenf.save_weights(fpath + '/weights/iter%04d'%(i//saveiter))
