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

sys.path.append('../../galference/utils/')
sys.path.append('../src/')
sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')
sys.path.append('../../hmc/src/')

from pmfuncs import Evolve
import tools
from pyhmc import PyHMC, PyHMC_batch, DualAveragingStepSize
from callback import callback_sampling, datafig
import recon

#$#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bs', type=float, default=200., help='box size')
parser.add_argument('--nc', type=int, help='mesh size')
parser.add_argument('--eps', type=float,  help='step size')
parser.add_argument('--nchains', type=float,  help='step size')
parser.add_argument('--jobid', type=int, help='an integer for the accumulator')
parser.add_argument('--suffix', type=str, help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--optinit', type=int, default=1, help="initialize near MAP")
parser.add_argument('--scaleprior', type=int, default=1, help="add power to scale to prior")
parser.add_argument('--nsamples', type=int, default=5000, help="number of HMC samples")
parser.add_argument('--truecosmo', type=int, default=1, help="use same cosmology for prior and data")
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--dnoise', type=float, default=1., help='noise level, 1 is shot noise')

args = parser.parse_args()
device = args.jobid
print(device, rank)

##########
nchains = 4
reconiter = 100
#burnin = 200
#mcmciter = 500
#ntrain = 10
tadapt = 50
thinning = 20
lpsteps1, lpsteps2 = 25, 50
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
donbody = False
order = 1
shotnoise = bs**3/nc**3
dnoise = args.dnoise


if order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA'%(bs, nc)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)


# Compute necessary Fourier kernels                                                              
evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)
if args.truecosmo == 1:
    evolvedata = evolve
else:
    cosmodata = flowpm.cosmology.Planck15().to_dict()
    cosmodata['Omega_c'] *= 1.05
    cosmodata['sigma8'] *= 0.95
    evolvedata = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order, cosmodict=cosmodata)


##################
##Generate DATA

np.random.seed(100)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc)
ic = evolvedata.z_to_lin(zic).numpy()
fin = evolvedata.pm(tf.constant(ic)).numpy()
data = fin + noise
data = data.astype(np.float32)
print("data shape : ", data.shape)
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
def unnormalized_log_prob(z, noiselevel=tf.constant(1.)):
    #Chisq
    final_field = evolve.pmz(z)
    residual = (final_field - data)/dnoise/noiselevel
    base = residual
    chisq = tf.multiply(base, base)
    chisq = 0.5 * tf.reduce_sum(chisq, axis=(-3, -2, -1))
    logprob = -chisq
    #Prior
    logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(z), axis=(-3, -2, -1)) 
    toret = logprob + logprior
    return  toret


@tf.function
def grad_log_prob(linear, noiselevel=tf.constant(1.)):
    with tf.GradientTape() as tape:
        tape.watch(linear)
        logposterior = tf.reduce_sum(unnormalized_log_prob(linear, noiselevel))
    grad = tape.gradient(logposterior, linear)
    return grad


########################################
####Setup


print("\nSetting up HMC\n")
start = time.time()
#######Correct log prob
py_log_prob = lambda x: unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
hmckernel = PyHMC_batch(py_log_prob, py_grad_log_prob, returnV=True)
stepsize = np.ones(nchains)*args.eps
epsadapt = DualAveragingStepSize(stepsize)
samples, pyacc = [], []


###################################################################################
###################################################################################
###########Search for good initialization
start = time.time()
seed = 100*device+rank + 2021
np.random.seed(seed)
x0 = np.random.normal(size=nchains*nc**3).reshape(nchains, nc, nc, nc).astype(np.float32)
if args.optinit == 1:
    print("Search MAP for good initialization")
    for iR, RR in enumerate(allR):
        x0 = recon.map_estimate(evolve, x0*0.1, data, RR, maxiter=reconiter)
        #minicz2 = z_to_lin(linearz).numpy().reshape(fin.shape)
        fig = callback_sampling([evolve.z_to_lin(x0)], ic, bs)
        plt.savefig(fpath + '/figs/map%02d-%02d'%(device, RR*10))
        plt.close()
        print("time taken for MAP estimate for R=%d : "%RR, time.time() - start)

#Add white noise
q = x0.copy()
if (args.optinit == 1) & (args.scaleprior==1):
    for i in range(q.shape[0]):
        print("Mean : ", x0[i].mean())
        if abs(x0[i].mean()) <1e-1: 
            k, pz = tools.power(x0[i]+1., boxsize = bs)
        else: 
            k, pz = tools.power(x0[i], boxsize = bs)
        pdiff = (bs/nc)**3 - pz
        xx, yy = k[pdiff > 0], pdiff[pdiff > 0]
        try:
            ipkdiff = lambda x: 10**np.interp(np.log10(x), np.log10(xx), np.log10(yy))
            q[i] +=  flowpm.linear_field(nc, bs, ipkdiff, seed=(seed+i)).numpy()[0]
        except Exception as e:
            print("Exception in upscaling power to prior : ", e)
            print("pdiff : ", pdiff)
            print("Not upscaling power")
            #ipkdiff = lambda x: 10**np.interp(np.log10(x), np.log10(k), np.log10(pz*0 + 0.2*(bs/nc)**3))
            #q[i] +=  flowpm.linear_field(nc, bs, ipkdiff, seed=(seed+i)).numpy()[0]


###################################################################################
print("\nStart sampling\n")
for i in range(args.nsamples):
    
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    q, _, acc, energy, _ = hmckernel.hmc_step(q.copy(), lpsteps, stepsize)
    prob = np.exp(energy[0] - energy[1])
    print('Accept/Reject in device %d for iteratiion %d : '%(device, i), list(zip(acc, prob)))
    if i < tadapt:
        prob[np.isnan(prob)] = 0.
        prob[prob > 1] = 1.
        stepsize, avgstepsize = epsadapt.update(prob)
        print("stepsizes : ", ["%0.3f"%i for i in stepsize.flatten()])
        print("Avg stepsizes : ", ["%0.3f"%i for i in avgstepsize.flatten()])
    elif i == tadapt:
        _, stepsize = epsadapt.update(prob)
        print("Step size fixed to : ", stepsize)
        np.save(fpath + '/stepsizes%d-%02d'%(device, rank), stepsize)

    #append
    pyacc.append(acc)
    if (i%thinning) == 0:
        samples.append(q.astype(np.float32)) 
        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
        print("Acceptance in device %d, rank %d = "%(device, rank), list(zip(np.unique(pyacc), np.unique(pyacc, return_counts=True)[1]/np.array(pyacc).size)))
        np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
        np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
        fig = callback_sampling([evolve.z_to_lin(i) for i in samples[-2:]], ic, bs)
        plt.savefig(fpath + '/figs/iter%02d-%05d'%(device, i))


##########
samples = np.array(samples)
np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
end = time.time()
print('Time taken in rank %d= '%rank, end-start)




