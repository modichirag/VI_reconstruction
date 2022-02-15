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
parser.add_argument('--jobid', type=int, default=0, help='an integer for the accumulator')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--repvi', type=int, default=0, help="reparametrization gradient")
parser.add_argument('--viwts', type=int, default=0, help="use k weights")
parser.add_argument('--mode', type=str, default="classic", help='mode for trenf')

args = parser.parse_args()
device = args.jobid
suffix = args.suffix


import sys, os, math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import scipy.optimize as sopt

sys.path.append('../../galference/utils/')
import tools
import diagnostics as dg

sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')

import flowpm
from astropy.cosmology import Planck15
# from flowpm.tfpm import PerturbationGrowth
#from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

sys.path.append('../src/')
#import flow_utils as futils
#import flows
#TransferSpectra = futils.TransferSpectra
#SimpleRQSpline = futils.SimpleRQSpline

sys.path.append('../../hmc/src/')
from pmfuncs import Evolve
from trenfmodel import VItrenf

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


evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     
kwts = (evolve.kmesh**2).copy()
kwts[0, 0, 0] = kwts[0, 0, 1]
kwts /= kwts[0, 0, 0] 
#kwts = kwts**0.5
#klin = np.loadtxt('../../galference/data/Planck15_a1p00.txt').T[0]
#plin = np.loadtxt('../../galference/data//Planck15_a1p00.txt').T[1]
#ipklin = iuspline(klin, plin)


##############################################


print("\nFor seed : ", args.seed)
np.random.seed(args.seed)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc)
ic = evolve.z_to_lin(zic).numpy()
fin = evolve.pm(tf.constant(ic)).numpy()
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
def likelihood(data, z, noiselevel=tf.constant(1.)):
    final_field = evolve.pmz(z)
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




@tf.function
def bbvi_trenf_wts(model, likelihood, data, Rsm, nsamples = tf.constant(1)):   
    
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
    gradzc = r2c3d(gradz)
    gradzc = gradzc / kwts
    gradz = c2r3d(gradzc)
    grad = tape.gradient(sample, model.trainable_variables, gradz)
    return tf.stack([logl, logp, logq, -elbo]), grad


def optimizetrenf(model, data, opt,  Rsm, niter=100, callback=callback, citer=50, nsamples=1, saveiter=100, repvi=args.repvi):
    
    losses = []
    spath = fpath + 'R%02d/'%(Rsm*10)
    os.makedirs(spath, exist_ok=True)
    os.makedirs(spath + '/opt', exist_ok=True)
    print('\nDoing for smoothing of R=%d\n'%Rsm)
    start = time.time()
    for epoch in range(niter+1):
        #print(epoch)
        if epoch%10 == 0: print(epoch)
        if repvi : 
            if epoch == 0: print("Using reparameterized gradients")
            loss, grad = bbvi_trenf_rp(model, likelihood, data, Rsm, nsamples=tf.constant(nsamples))
        if args.viwts : 
            if epoch == 0: print("Using k weighted loss")
            loss, grad = bbvi_trenf_wts(model, likelihood, data, Rsm, nsamples=tf.constant(nsamples))
        else: loss, grad = bbvi_trenf(model, likelihood, data, Rsm, nsamples=tf.constant(nsamples))
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


vitrenf = VItrenf(nc, nlayers=3, evolve=evolve, mode=args.mode)
for i in vitrenf.trainable_variables:
    print(i.name, i.shape)

plt.imshow(vitrenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig(fpath + 'initsample')
plt.colorbar()
plt.close()

print("\nStart VI\n")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-2,
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
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #dataR = gauss_smooth(data, R)
    losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 5001, callback, citer=100)

#R = 0.
#losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 1001, callback, citer=10)
    
#opt = tf.keras.optimizers.Adam(learning_rate=0.01)
#losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(0.), 10001, callback, citer=100)

