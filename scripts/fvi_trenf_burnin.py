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
parser.add_argument('--eps', type=float, default=0.001, help='step size')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--scaleprior', type=int, default=1, help="add power to scale to prior")
parser.add_argument('--nchains', type=int, default=4, help="number of chains")
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--reconiter', type=int, default=100, help="number of iterations for reconstruction")
parser.add_argument('--burnin', type=int, default=200, help="number of iterations for burnin")
parser.add_argument('--tadapt', type=int, default=50, help="number of iterations for eps adaptation")
parser.add_argument('--ntrain', type=int, default=10, help="number of training iterations")
parser.add_argument('--thinning', type=int, default=20, help="thinning")
parser.add_argument('--lpsteps1', type=int, default=25, help="min leapfrog steps")
parser.add_argument('--lpsteps2', type=int, default=50, help="max leapfrog steps")
parser.add_argument('--mcmciter', type=int, default=200, help="number of only mcmc iterations")
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")
parser.add_argument('--annealorder', type=float, default=3., help="order to anneal in flow step")
#
parser.add_argument('--mode', type=str, default="classic", help='mode for trenf')
parser.add_argument('--nlayers', type=int, default=3, help='number of trenf layers')
parser.add_argument('--nbins', type=int, default=32, help='number of trenf layers')
parser.add_argument('--nknots', type=int, default=100, help='number of trenf layers')
parser.add_argument('--linknots', type=int, default=0, help='linear spacing for knots')
parser.add_argument('--fitnoise', type=int, default=1, help='fitnoise')
parser.add_argument('--fitscale', type=int, default=1, help='fitscale')
parser.add_argument('--fitmean', type=int, default=1, help='fitmean')
parser.add_argument('--meanfield', type=int, default=1, help='meanfield for affine')
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
from pmfuncs import Evolve
from pyhmc import PyHMC, PyHMC_batch, DualAveragingStepSize
#from trenfmodel import VItrenf
import trenfmodel
import recon
from callback import callback, datafig, callback_fvi, callback_sampling

##########
nchains = args.nchains
reconiter = args.reconiter
burnin = args.burnin
mcmciter = args.mcmciter
ntrain = args.ntrain
tadapt = args.tadapt
thinning = args.thinning
lpsteps1, lpsteps2 = args.lpsteps1, args.lpsteps2
recalib = 1
#
if args.debug:
    reconiter = 10
    burnin = 10
    mcmciter = 10
    tadapt = 10
    thinning = 2 
    lpsteps1, lpsteps2 = 3, 5

probjump = 0.3
allRs = [0., 1., 2., 4.]
allR = allRs[:args.nR + 1][::-1]

#
bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
nsims = 200
donbody = False
order = args.order
shotnoise = bs**3/nc**3
dnoise = 1. #shotnoise/nc**1.5  
if donbody: fpath = '/mnt/ceph/users/cmodi/galference/dm_fvi/L%04d_N%04d_T%02d'%(bs, nc, nsteps)
elif order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_fvi/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_fvi/L%04d_N%04d_ZA'%(bs, nc)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)
os.makedirs('%s'%fpath + '/opt/', exist_ok=True)
os.makedirs('%s'%fpath + '/weights/', exist_ok=True)
os.makedirs('%s'%fpath + '/burnin/', exist_ok=True)


klin = np.loadtxt('../../galference/data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../galference/data//Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     


##############################################
##Generate DATA
print("\nFor seed : ", args.seed)
np.random.seed(args.seed)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
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

    


##############################################

@tf.function
def unnormalized_log_prob(z):
    #z = tf.reshape(z, data.shape)
    #Chisq
    final_field = evolve.pmz(z)
    logprob = tfd.Normal(final_field, scale=dnoise).log_prob(tfdata)
    logprob = tf.reduce_sum(logprob, axis=[-3, -2, -1])
    logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(z), axis=[-3, -2, -1])
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
hmckernel = PyHMC_batch(py_log_prob, py_grad_log_prob, returnV=True)
epsadapt = DualAveragingStepSize(args.eps)


def flow_step(model, x, lpx):
    nsamples = x.shape[0]
    lqx = model.q.log_prob(x)
    
    y = model.q.sample(nsamples)*1.
    lpy = unnormalized_log_prob(y)
    lqy = model.q.log_prob(y)
    print(tf.stack([lpx, lpy, lqx, lqy], axis=0).numpy())
    prob = tf.exp((lpy - lpx)/nc**args.annealorder + lqx - lqy)
    #prob = tf.exp((lpy - lpx + lqx - lqy))

    accept = tf.where(tf.random.uniform([1]) <= tf.minimum(1., prob))
    reject = tf.where(tf.random.uniform([1]) > tf.minimum(1., prob))
    z = tf.scatter_nd(accept, tf.gather_nd(y, accept), y.shape) + \
         tf.scatter_nd(reject, tf.gather_nd(x, reject), y.shape)

    lpz = tf.where(tf.random.uniform([1]) <= tf.minimum(1., prob), lpy, lpx)
    acc = tf.where(tf.random.uniform([1]) <= tf.minimum(1., prob), tf.ones(lpx.shape)*11, tf.ones(lpx.shape)*10.)
    
    return z, lpz, prob, acc, tf.constant(1.)


def mcmc_step(q, stepsize):
    #stepsize = np.random.uniform(0.01, 0.02, 1)
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    q, _, acc, prob, _ = hmckernel.hmc_step(q.numpy(), lpsteps, stepsize)
    lpq = tf.constant(prob[2]*-1.)
    prob = np.exp(prob[0] - prob[1])
    q = tf.constant(q, dtype=tf.float32)
    return q, lpq, tf.constant(prob), tf.constant(acc*1.), tf.constant(0.)


#@tf.function
def fvi(model, q, logpq, i, stepsize):   
    
    elbos = []
    nsamples = 1
    start = time.time()
    if i < mcmciter:
        print("MCMC only")
        q, logpq, prob, acc, jump =  mcmc_step(q, stepsize)
    else:    
        q, logpq, prob, acc, jump = tf.cond(tf.random.uniform([1]) > probjump, \
                                      lambda : mcmc_step(q, stepsize), \
                            lambda : flow_step(model, q, logpq))
        if jump == 1:
            for _ in range(recalib):
                print("Jump, recaliberate")
                q, logpq, prob2, acc2, jump2 =  mcmc_step(q, stepsize)
    print("prob ", prob.numpy(), "to accept/reject with ", acc.numpy(), " in time %0.3f"%(time.time() - start))

    return q, logpq, acc, jump


@tf.function
def fvi_train(model, samples, opt):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        #q, logpq, acc = fvi_step(model, q, logpq)
        neglogq = - tf.reduce_mean(model.q.log_prob(samples))
        if args.regwt0 != 0:
            #logq2 = - tf.reduce_mean(model.q.log_prob(samples2))
            #samplevi = model.sample(samples.shape[0])*1.
            #logqvi = - tf.reduce_mean(model.q.log_prob(samplevi))
            #reg = regwt * tf.reduce_mean((logq2 - logqvi)**2.)
            #loss = neglogq + reg
            loss = neglogq
        else: 
            loss = neglogq
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return neglogq




def optimizetrenf(model, data, opt,  Rsm, niter=100, callback=callback, citer=50, nsamples=1, saveiter=100, sample=None, batch_size=3, stepsize=0.01):
    
    if sample is None: 
        sample = tf.random.normal([nsamples, nc, nc, nc])
    logpsample = unnormalized_log_prob(sample)
        
    accs, samples,losses = [], [], []
    samples.append(sample)
    os.makedirs(fpath + '/opt', exist_ok=True)
    start = time.time()
    #dry run
    sample, logpsample, acc, jump= fvi(model, sample, logpsample, 0, stepsize)
    accs.append(acc)
    samples.append(sample)
    for epoch in range(niter+1):
        print(epoch)
        sample, logpsample, acc, jump = fvi(model, sample, logpsample, epoch, stepsize)
        accs.append(acc)
        samples.append(sample)
        for itrain in range(min(ntrain, int(ntrain*(2*epoch/mcmciter)))):
            if jump == 1: 
                print("Jump, no train")
                break
            idx = np.arange(len(samples))
            try: 
                #idxsample = np.random.choice(idx, batch_size, p=idx/idx.sum(), replace=False)
                idxsample = np.random.choice(idx, batch_size, replace=False)
            except: 
                idxsample = np.random.choice(idx, batch_size, replace=True)
            #idxsample = [-1] + list(idxsample)
            idlast = min(-1 * np.random.choice(ntrain), -1)
            idxsample = [idlast] + list(idxsample)
            print("idx :", idxsample)
            batch = tf.concat([samples[i] for i in idxsample], axis=0)
            loss = fvi_train(model, batch, opt)
            losses.append(loss/nc**3)
            print("LogL, Logp, Logq, ELBO at epoch %d is: "%epoch, loss.numpy())

        if epoch%citer == 0: 
            print("Time taken for %d iterations : "%citer, time.time() - start)
            print("Accept counts : ", np.unique(accs, return_counts=True))
            #
            fig = callback_fvi(model, ic, bs, losses)
            plt.savefig(fpath + '/figs/iter%05d'%epoch)
            plt.close()
            #
            fig = callback_sampling([evolve.z_to_lin(i) for i in samples[-2:]], ic, bs)
            plt.savefig(fpath + '/figs/samples%05d'%epoch)
            plt.close()
            np.save(fpath + 'loss', np.array(losses))
            np.save(fpath + 'samples', np.array(samples))
            np.save(fpath + 'accepts', np.array(accs))
            start = time.time()
        if epoch%saveiter == 0: 
            model.save_weights(fpath + '/weights/iter%04d'%(epoch//saveiter))
            np.save(fpath + '/opt/iter%04d'%(epoch//saveiter), opt.get_weights(), allow_pickle=True)
            #print('Weights saved')
            
    return losses 




###################################################################################
###################################################################################
###########Search for good initialization
start = time.time()
seed = 100*device+rank + 2021
np.random.seed(seed)
x0 = np.random.normal(size=nchains*nc**3).reshape(nchains, nc, nc, nc).astype(np.float32)
#$#for iR, RR in enumerate(allR):
#$#    x0 = recon.map_estimate(evolve, x0, data, RR, maxiter=reconiter)
#$#    #minicz2 = z_to_lin(linearz).numpy().reshape(fin.shape)
#$#    fig = callback_sampling([evolve.z_to_lin(x0)], ic, bs)
#$#    plt.savefig(fpath + '/figs/map%02d-%02d'%(device, RR*10))
#$#    plt.close()
#$#    print("time taken for MAP estimate for R=%d : "%RR, time.time() - start)
#$#
#$##Add white noise
#$#q = x0.copy()
#$#if args.scaleprior:
#$#    for i in range(q.shape[0]):
#$#        if abs(x0[i].mean()) <1e-1: 
#$#            k, pz = tools.power(x0[i]+1., boxsize = bs)
#$#        else: 
#$#            k, pz = tools.power(x0[i], boxsize = bs)
#$#        pdiff = (bs/nc)**3 - pz
#$#        xx, yy = k[pdiff > 0], pdiff[pdiff > 0]
#$#        try:
#$#            ipkdiff = lambda x: 10**np.interp(np.log10(x), np.log10(xx), np.log10(yy))
#$#            q[i] +=  linear_field(nc, bs, ipkdiff, seed=seed).numpy()[0]
#$#        except Exception as e:
#$#            print(e)
#$#            ipkdiff = lambda x: 10**np.interp(np.log10(x), np.log10(k), np.log10(pz*0 + 0.2*(bs/nc)**3))
#$#            q[i] +=  linear_field(nc, bs, ipkdiff, seed=seed).numpy()[0]
#$#
#$#
#########################################
###########DO HMC to reach the typical set
#$#
#$#
#$#print("\nStart BurnIN\n")
#$#samples, pyacc = [], []
#$#start = time.time()
#$#stepsize = np.ones(nchains)*args.eps
#$#epsadapt = DualAveragingStepSize(stepsize)
#$#print(q.shape)
#$#
#$#for i in range(burnin):    
#$#    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
#$#    q, lpq, prob, acc = mcmc_step(tf.constant(q),stepsize)
#$#    #q, _, acc, energy, _ = hmckernel.hmc_step(q.copy(), lpsteps, stepsize)
#$#    q, prob, acc = q.numpy(), prob.numpy(), acc.numpy()
#$#    print('Accept/Reject in device %d for iteratiion %d : '%(device, i), list(zip(acc, prob)))
#$#    if i < tadapt:
#$#        prob[np.isnan(prob)] = 0.
#$#        prob[prob > 1] = 1.
#$#        stepsize, avgstepsize = epsadapt.update(prob)
#$#        print("stepsizes : ", ["%0.3f"%i for i in stepsize.flatten()])
#$#        print("Avg stepsizes : ", ["%0.3f"%i for i in avgstepsize.flatten()])
#$#    elif i == tadapt:
#$#        _, stepsize = epsadapt.update(prob)
#$#        print("Step size fixed to : ", stepsize)
#$#        np.save(fpath + '/stepsizes%d-%02d'%(device, rank), stepsize)
#$#
#$#    #append
#$#    pyacc.append(acc)    
#$#    if (i%thinning) == 0:
#$#        samples.append(q.astype(np.float32)) 
#$#        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
#$#        print("Acceptance in device %d, rank %d = "%(device, rank), list(zip(np.unique(pyacc), np.unique(pyacc, return_counts=True)[1]/np.array(pyacc).size)))
#$#        np.save(fpath + '/burnin/samples%d-%02d'%(device, rank), np.array(samples))
#$#        np.save(fpath + '/burnin/accepts%d-%02d'%(device, rank), np.array(pyacc))
#$#        fig = callback_sampling([evolve.z_to_lin(i) for i in samples[-2:]], ic, bs)
#$#        plt.savefig(fpath + '/figs/burnin%02d-%05d'%(device, i))
#$#

################################################################
###################VI
print("\nStart VI\n")
vitrenf = trenfmodel.VItrenf(nc, nlayers=args.nlayers, evolve=evolve, nbins=args.nbins, nknots=args.nknots, mode=args.mode, \
                             linknots=bool(args.linknots), fitnoise=bool(args.fitnoise), fitscale=bool(args.fitscale), \
                             fitmean=bool(args.fitmean), meanfield=bool(args.meanfield))
for i in vitrenf.variables:
    print(i.name, i.shape)

plt.imshow(vitrenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig(fpath + 'initsample')
plt.colorbar()
plt.close()

#sample = tf.constant(np.random.normal(size=nchains*nc**3).reshape(nchains, nc, nc, nc).astype(np.float32))
#stepsize = np.ones(4)*0.01
truesamplespath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA-kwts4-fourier/'%(bs, nc)
samples = []
stepsizes = []
for i in range(nchains):
    zk = np.load(truesamplespath + '/samples%d-00.npy'%i)[-1].astype(np.float32)
    print("zk shape ", zk.shape)
    samples.append(evolve.zk_to_z(tf.constant(zk).numpy()))
    stepsizes.append(np.load(truesamplespath + '/stepsizes%d-00.npy'%i)[0].astype(np.float32))
    #truelps.append(np.load(truesamplespath + '/lps%d-00.npy'%i)[:, 0])
sample = tf.constant(np.concatenate(samples), dtype=tf.float32)
stepsize = np.array(stepsizes)
print("sample shape ", sample.shape)
print("stepsize : ", stepsize)
 

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-2,
    decay_steps=200,
    decay_rate=0.95,
    staircase=True)

R = 0.
opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)
losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 5001, callback, citer=20, sample=sample, stepsize=stepsize)

