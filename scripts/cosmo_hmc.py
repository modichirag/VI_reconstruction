#from imports import *
import numpy as np
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import sys, os, time, argparse
import flowpm
print(flowpm, "\n")

sys.path.append('../../galference/utils/')
sys.path.append('../src/')
sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')
sys.path.append('../../hmc/src/')

from pmfuncs import Evolve
import tools
from pyhmc import PyHMC, PyHMC_batch, DualAveragingStepSize
from callback import callback_sampling, datafig
import recon
from flowpm.scipy.interpolate import interp_tf

#$#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bs', type=float, default=200., help='box size')
parser.add_argument('--nc', type=int, help='mesh size')
parser.add_argument('--eps', type=float,  help='step size')
parser.add_argument('--nchains', type=int,  help='nchains')
parser.add_argument('--jobid', type=int, help='an integer for the accumulator')
parser.add_argument('--suffix', type=str, help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--optinit', type=int, default=1, help="initialize near MAP")
parser.add_argument('--scaleprior', type=int, default=1, help="add power to scale to prior")
parser.add_argument('--nsamples', type=int, default=5000, help="number of HMC samples")
parser.add_argument('--truecosmo', type=int, default=1, help="use same cosmology for prior and data")
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--dnoise', type=float, default=1., help='noise level, 1 is shot noise')
parser.add_argument('--reconiter', type=int, default=100, help="number of iterations for reconstruction")
parser.add_argument('--burnin', type=int, default=200, help="number of iterations for burnin")
parser.add_argument('--tadapt', type=int, default=50, help="number of iterations for eps adaptation")
parser.add_argument('--ntrain', type=int, default=10, help="number of training iterations")
parser.add_argument('--thinning', type=int, default=20, help="thinning")
parser.add_argument('--lpsteps1', type=int, default=10, help="min leapfrog steps")
parser.add_argument('--lpsteps2', type=int, default=20, help="max leapfrog steps")
parser.add_argument('--mcmciter', type=int, default=200, help="number of only mcmc iterations")
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")
parser.add_argument('--donbody', type=int, default=0, help="Do Nbody or ZA/LPT")




args = parser.parse_args()
device = args.jobid
print(device, rank)

##########
nchains = args.nchains
reconiter = args.reconiter
burnin = args.burnin
mcmciter = args.mcmciter
ntrain = args.ntrain
tadapt = args.tadapt
thinning = args.thinning
lpsteps1, lpsteps2 = args.lpsteps1, args.lpsteps2
#

allRs = [0., 1., 2., 4.]
allR = allRs[:args.nR + 1][::-1]

if args.debug == 1:
    nchains = 2
    reconiter = 10
    burnin = 10
    mcmciter = 10
    tadapt = 10
    thinning = 2 
    lpsteps1, lpsteps2 = 3, 5

suffix = args.suffix
bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
donbody = args.donbody
order = args.order
shotnoise = bs**3/nc**3
dnoise = args.dnoise


if donbody: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc/L%04d_N%04d_T%02d'%(bs, nc, nsteps)
elif order == 2: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc/L%04d_N%04d_ZA'%(bs, nc)
elif order == 0: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc/L%04d_N%04d_IC'%(bs, nc)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)


# Compute necessary Fourier kernels                                                              
evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)
if args.truecosmo == 1:
    evolvedata = evolve
else:
    #cosmodata = flowpm.cosmology.Planck15().to_dict()
    #cosmodata['Omega_c'] *= 1.05
    #cosmodata['sigma8'] *= 0.95
    #evolvedata = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order, cosmodict=cosmodata)
    evolvedata = evolve

cosmodata = evolve.cosmodict
##################
##Generate DATA

np.random.seed(100)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc) 
ic = evolvedata.z_to_lin(zic).numpy()
if order == 0: fin = ic.copy()
else: fin = evolvedata.pm(tf.constant(ic)).numpy()
data = fin + noise
data = data.astype(np.float32)
params = tf.stack([cosmodata['Omega_c'], cosmodata['sigma8'], cosmodata['Omega_b'], cosmodata['h']])
params = tf.stack([cosmodata['Omega_c'], cosmodata['sigma8'], cosmodata['h']])
print("params : ", params)

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
def whitenoise_to_linear(evolve, whitec, pk):
    nc, bs = evolve.nc, evolve.bs
    pkmesh = pk(evolve.kmesh)
    lineark = tf.multiply(whitec, tf.cast((pkmesh /bs**3)**0.5, whitec.dtype))
    linear = flowpm.utils.c2r3d(lineark, norm=nc**3)
    return linear


@tf.function
def cosmo_sim(params, whitec):
    cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1], h=params[2])
    k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
    pk = flowpm.tfpower.linear_matter_power(cosmology, k)
    pk_fun = lambda x: tf.cast(
        tf.reshape(interp_tf(
                tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.complex64)
    ic = whitenoise_to_linear(evolve, whitec, pk_fun)
    if order == 0: return ic
    final_field = evolve.pm(ic, cosmology.to_dict())
    return final_field


@tf.function
def unnormalized_log_prob(params,  whitec):
    with tf.GradientTape() as tape:
        tape.watch(params)
        final_field = cosmo_sim(params, whitec)
        loglik = -tf.reduce_sum(((data- final_field)/dnoise)**2)
        logprob = tfd.Uniform(0.15, 0.35).log_prob(params[0]) + tfd.Uniform(0.65, 0.95).log_prob(params[1])
        logprob += tfd.Uniform(0.55, 0.80).log_prob(params[2])
        loss = loglik + logprob
    return loss

@tf.function
def grad_log_prob(params,  whitec):
    with tf.GradientTape() as tape:
        tape.watch(params)
        final_field = cosmo_sim(params, whitec)
        loss = -tf.reduce_sum((data- final_field)**2)
    return tape.gradient(loss, params)


@tf.function
def val_and_grad(params,  whitec):
    with tf.GradientTape() as tape:
        tape.watch(params)
        final_field = cosmo_sim(params, whitec)
        loss = tf.reduce_sum((data- final_field)**2)
    return loss, tape.gradient(loss, params)


def corner(samples, pp):
    ndim = samples.shape[1]
    fig, ax = plt.subplots(ndim, ndim, figsize=(ndim*2, ndim*2), sharex=False, sharey=False)
    for i in range(ndim):
        for j in range(ndim):
            if i == j: 
                ax[i, j].hist(samples[:, i], bins='auto', density=True)
                ax[i, j].axvline(samples[:, i].mean(), ls="--", color="r", label=pp[i])
                ax[i, j].axvline(samples[:, i].mean()+samples[:, i].std(), ls="--", color="r")
                ax[i, j].axvline(samples[:, i].mean()-samples[:, i].std(), ls="--", color="r")
                ax[i, j].axvline(pp[i], ls="--", color="k")
                ax[i, j].legend()
                #             ax[i, j].set_title(names[i])
            elif i>j: 
                ax[i, j].plot(samples[:, i], samples[:, j], '.')
                ax[i, j].axvline(pp[i], ls="--", color="k")
                ax[i, j].axhline(pp[j], ls="--", color="k")
                ax[i, j].axvline(samples[:, i].mean(), ls="--", color="r")
                ax[i, j].axhline(samples[:, j].mean(), ls="--", color="r")
            else: ax[i, j].set_visible(False)
    plt.tight_layout()
    return fig

################################################################################
################################################################################
np.random.seed(2021 + device*100)
whitec = flowpm.utils.r2c3d(tf.constant(zic, dtype=tf.float32)* nc**1.5)

model, gg = val_and_grad(params, whitec)
params0 = tf.Variable(np.array([0.22, 0.85, 0.6]).astype(np.float32)*np.random.uniform(0.9, 1.1))
print(params, params0)
opt = tf.keras.optimizers.Adam(0.01)

losses = []
pp = []
for i in range(200):
    l, grad = val_and_grad(params0, whitec)
    opt.apply_gradients(zip([grad], [params0]))
    losses.append(l)
    pp.append(params0.numpy())
    print(pp[-1])

print("True value : ", params)
print("MAP estiamte :  ", pp[-1])
fig, ax = plt.subplots(1, 4, figsize=(13, 4))
ax[0].plot(np.array(pp)[:, 0])
ax[0].axhline(params[0])
ax[1].plot(np.array(pp)[:, 1])
ax[1].axhline(params[1])
ax[2].plot(np.array(pp)[:, 2])
ax[2].axhline(params[2])
ax[3].plot(losses)
ax[3].semilogy()
plt.savefig(fpath + 'map.png')


########################################
####Setup


print("\nSetting up HMC\n")
start = time.time()
#######Correct log prob
py_log_prob = lambda x: unnormalized_log_prob(tf.constant(x, dtype=tf.float32), whitec).numpy().astype(np.float32)
py_grad_log_prob = lambda x: grad_log_prob(tf.constant(x, dtype=tf.float32), whitec).numpy().astype(np.float32)
hmckernel = PyHMC(py_log_prob, py_grad_log_prob, returnV=True)
stepsize = args.eps
epsadapt = DualAveragingStepSize(stepsize)
samples, pyacc = [], []
q = params0.numpy()

###################################################################################
print("\nStart sampling\n")
for i in range(args.nsamples):
    
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    q, _, acc, energy, _ = hmckernel.hmc_step(q.copy(), lpsteps, stepsize)
    prob = np.exp(energy[0] - energy[1])
    #print('Accept/Reject in device %d for iteratiion %d : '%(device, i), list(zip(acc, prob)))
    print('Accept/Reject in device %d for iteratiion %d : '%(device, i), acc, prob)
    if i < tadapt:
        if np.isnan(prob): prob = 0.
        if prob > 1: prob = 1.

        stepsize, avgstepsize = epsadapt.update(prob)
        print(stepsize, avgstepsize)
    elif i == tadapt:
        _, stepsize = epsadapt.update(prob)
        print("Step size fixed to : ", stepsize)
        np.save(fpath + '/stepsizes%d-%02d'%(device, rank), stepsize)

    #append
    pyacc.append(acc)
    samples.append(q.astype(np.float32)) 
    if (i%thinning) == 0:
        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
        print("Acceptance in device %d, rank %d = "%(device, rank), list(zip(np.unique(pyacc), np.unique(pyacc, return_counts=True)[1]/np.array(pyacc).size)))
        np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
        np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
        print(np.array(samples).shape)
        
        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        npsamples = np.array(samples)
        fig = corner(npsamples, params.numpy())
        plt.savefig(fpath + '/figs/iter%02d-%05d'%(device, i))
        plt.close()

##########
samples = np.array(samples)
np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
end = time.time()
print('Time taken in rank %d= '%rank, end-start)
