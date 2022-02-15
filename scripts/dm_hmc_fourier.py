import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import tensorflow_probability as tfp
tfd = tfp.distributions

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import sys, os, time, argparse
import flowpm


import sys, os, time
sys.path.append('../../galference/utils/')
sys.path.append('../src/')
sys.path.append('/mnt/home/cmodi/Research/Projects/flowpm-pgd')

import tools
import diagnostics as dg
from pmfuncs import Evolve
from pyhmc import PyHMC, DualAveragingStepSize
from callback import callback_sampling, datafig
from hmcfuncs import DM_config, DM_fourier, Kwts
import recon

##########
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--eps', type=float,  help='step size')
parser.add_argument('--nc', type=int, help='mesh size')
parser.add_argument('--bs', type=float, help='box size')
parser.add_argument('--jobid', type=int, help='an integer for the accumulator')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')
parser.add_argument('--nR', type=int, default=0, help="number of smoothings")
parser.add_argument('--scaleprior', type=int, default=1, help="add power to scale to prior")
parser.add_argument('--optinit', type=int, default=1, help="initialize near MAP")
parser.add_argument('--nsamples', type=int, default=5000, help="number of HMC samples")
parser.add_argument('--truecosmo', type=int, default=1, help="use same cosmology for prior and data")
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--dnoise', type=float, default=1., help='noise level, 1 is shot noise')
parser.add_argument('--Rmin', type=float, default=0., help='Rmin')
parser.add_argument('--kwts', type=int, default=0, help='use kwts')
parser.add_argument('--pamp', type=float, default=1., help='amplitude of initial power')


args = parser.parse_args()
device = args.jobid


##########
nchains = 1
reconiter = 100
#burnin = 200
#mcmciter = 500
#ntrain = 10
tadapt = 100
thinning = 20
lpsteps1, lpsteps2 = 25, 50
allRs = [args.Rmin, 1., 2., 4.]
allR = allRs[:args.nR + 1][::-1]

if args.debug == 1:
    nchains = 1
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
dnoise = 1 #shotnoise/nc**1.5  


if order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA'%(bs, nc)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)

# Compute necessary Fourier kernels                                                               
evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     

np.random.seed(100)
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
k, pic = tools.power(ic[0], boxsize=bs)
k, pf = tools.power(fin[0], boxsize=bs)
k, pd = tools.power(data[0], boxsize=bs)
k, pn = tools.power(1+noise[0], boxsize=bs)
knoise = evolve.kmesh[evolve.kmesh > k[(pn > pf)][0]].min()
print("Noise dominated after : ", knoise, (evolve.kmesh > knoise).sum()/nc**3)

##############################################
@tf.function
def unnormalized_log_prob(zk, noiselevel=tf.constant(1.)):
    #This function will not have a gradient due to RFFT
    logprior = unnormalized_log_prior(zk)
    link = evolve.zk_to_link(zk) 
    lin = tf.signal.irfft3d(tf.complex(link[..., 0], link[..., 1]))
    loglik = unnormalized_log_likelihood(lin, noiselevel)
    logprob = logprior + loglik
    return logprob

@tf.function
def unnormalized_log_prior(zk):
    zscale = 1. #(nc**3/2)**0.5
    logprior = tf.reduce_sum(tfd.Normal(0, zscale).log_prob(zk), axis=(-4, -3, -2, -1)) 
    return logprior

@tf.function
def unnormalized_log_likelihood(lin, noiselevel=tf.constant(1.)):
    #Chisq
    final_field = evolve.pm(lin)
    loglik = tf.reduce_sum(tfd.Normal(final_field, dnoise*noiselevel).log_prob(tfdata), axis=(-3, -2, -1))
    return loglik


@tf.function
def grad_log_prob(zk, noiselevel):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(zk)
        link = evolve.zk_to_link(zk) 
        logprior = unnormalized_log_prior(zk)
        
    lin = tf.signal.irfft3d(tf.complex(link[..., 0], link[..., 1]))
    
    with tf.GradientTape() as tape2:
        tape2.watch(lin)
        loglik = unnormalized_log_likelihood(lin)
        
    glin = tape2.gradient(loglik, lin)
    glink = tf.signal.rfft3d(glin)
    fac = np.ones(glink.shape)*2
    fac[:, :, :, 0] = 1.
    glink = glink * fac / nc**3
    #glink = 2 * tf.signal.rfft3d(glin) /nc**3  #2 because both + and -k contribute, nc**3 balances nc**1.5 in zk_to_link
    glinkz0, glinkz1 = tf.math.real(glink), tf.math.imag(glink)
    glinkz = tf.stack([glinkz0, glinkz1], -1)    
    gzlik = tape.gradient(link, zk, glinkz)
    gzprior = tape.gradient(logprior, zk)
    grad = gzlik + gzprior
    loss = logprior + loglik
    
    return grad
    


###################################################################################
###################################################################################
###########Search for good initialization
start = time.time()
seed = 100*device+rank + 2021
np.random.seed(seed)
x0 = np.random.normal(size=nchains*nc**3).reshape(nchains, nc, nc, nc).astype(np.float32) * args.pamp
if args.optinit == 1:
    for iR, RR in enumerate(allR):
        x0 = recon.map_estimate(evolve, x0, data, RR, maxiter=reconiter)
        fig = callback_sampling([evolve.z_to_lin(x0)], ic, bs)
        plt.savefig(fpath + '/figs/map%02d-%02d'%(device, RR*10))
        plt.close()
        print("time taken for Scipy LBFGS : ", time.time() - start)

#Add white noise
if (args.optinit == 1) & (args.scaleprior==1):
    print('\nScaling to prior')
    if abs(x0.mean()) <1e-1: k, pz = tools.power(x0[0]+1., boxsize = bs)
    else: k, pz = tools.power(x0[0], boxsize = bs)
    k = k[1:]
    pz = pz[1:]
    pdiff = (bs/nc)**3 - pz
    print(pz, pdiff)
    np.save(fpath + 'pdiff%02d'%device, pdiff)
    xx, yy = k[pdiff > 0], pdiff[pdiff > 0]
    ipkdiff = lambda x: 10**np.interp(np.log10(x), np.log10(xx), np.log10(yy))
    x0 = x0 + flowpm.linear_field(nc, bs, ipkdiff, seed=seed).numpy()
else:
    pass

#Generate q
xc = np.fft.rfftn(x0)/ nc**1.5 * 2**0.5 
x0 = np.stack([xc.real, xc.imag], -1).astype(np.float32)
#x0 = np.random.normal(0, 1, 2*nc*nc*(nc//2+1)).reshape(1, nc, nc, nc//2+1, 2).astype(np.float32) #2* 0.1*nc**1.5
q = x0.copy() 

if args.kwts == 1:
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**2
    kwts[0, 0, 0] = kwts[0, 0, 1]
    kwts /= kwts[0, 0, 0] 
    kwts = np.stack([kwts, kwts], axis=-1)
if args.kwts == 2:
    #ikwts
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**2
    kwts[0, 0, 0] = kwts[0, 0, 1]
    kwts /= kwts[0, 0, 0] 
    kwts = np.stack([kwts, kwts], axis=-1)
    kwts = 1/kwts
if args.kwts == 3:
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**3
    kwts[0, 0, 0] = kwts[0, 0, 1]
    kwts = np.stack([kwts, kwts], axis=-1)
if args.kwts == 4:
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**3
    kwts[0, 0, 0] = kwts[0, 0, 1]
    mask = kmesh > knoise
    kwts[mask] = knoise**3
    kwts = np.stack([kwts, kwts], axis=-1)
    kwts /= knoise**3
if args.kwts == 5:
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**3
    kwts[0, 0, 0] = kwts[0, 0, 1]
    mask = kmesh > knoise
    kwts[mask] = knoise**3
    kwts = np.stack([kwts, kwts], axis=-1)
    kwts /= knoise**3
    kwts = 1/kwts
if args.kwts == 6:
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**3
    kwts[0, 0, 0] = kwts[0, 0, 1]
    mask = kmesh > knoise
    kwts[mask] = knoise**3
    kwts = np.stack([kwts, kwts], axis=-1)
if args.kwts == 7:
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
    kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
    kwts = kmesh**3
    kwts[0, 0, 0] = kwts[0, 0, 1]
    mask = kmesh > knoise
    kwts[mask] = knoise**3
    kwts = np.stack([kwts, kwts], axis=-1)
    kwts /= knoise**3
    threshold=1e-3
    mask2 = kwts < threshold
    print(mask2.sum(), mask2.sum()/nc**3)
    kwts[mask2] = threshold
else: 
    kwts = None

################################################################################
################################################################################
####Sampling
print("\nstartng HMC in \n", device, rank, size)
start = time.time()

    
dmfuncs = DM_fourier(evolve, tfdata, dnoise=dnoise)
kwts = Kwts(evolve, mode=4, knoise=knoise)
py_log_prob = lambda x: dmfuncs.unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: dmfuncs.grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
#py_log_prob = lambda x: unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
#py_grad_log_prob = lambda x: grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
#hmckernel = PyHMC(py_log_prob, py_grad_log_prob, invmetric_diag=kwts)
epsadapt = DualAveragingStepSize(args.eps)
stepsize = args.eps
samples, pyacc = [], []

for i in range(args.nsamples):
    print(i)
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    q, _, acc, energy, _ = hmckernel.hmc_step(q, lpsteps, stepsize)
    prob = np.exp(energy[0] - energy[1])
    if acc == 1: print('Accept in device %d with %0.2f'%(device, prob), energy)
    else: print('Reject in device %d with %0.2f'%(device, prob), energy)
    if i < tadapt:
        if np.isnan(prob): prob = 0.
        if prob > 1: prob = 1.
        stepsize, avgstepsize = epsadapt.update(prob)
        print("stepsize and avg : ", stepsize, avgstepsize)
    elif i == tadapt:
        _, stepsize = epsadapt.update(prob)
        print("Step size fixed to : ", stepsize)
        np.save(fpath + '/stepsizes%d-%02d'%(device, rank), stepsize)

    #append
    pyacc.append(acc)
    if (i%thinning) == 0:
        samples.append(q.astype(np.float32)) 
        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
        print("Acceptance in device %d, rank %d = "%(device, rank), list(zip(np.unique(pyacc), np.unique(pyacc, return_counts=True)[1]/len(pyacc))))
        np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
        np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
        fig = callback_sampling([evolve.zk_to_lin(i) for i in samples[-10:]], ic, bs)
        plt.savefig(fpath + '/figs/iter%02d-%05d'%(device, i))


##########
samples = np.array(samples)
np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
end = time.time()
print('Time taken in rank %d= '%rank, end-start)
