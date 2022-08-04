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
from callback import callback_sampling, datafig, corner
import recon
from flowpm.scipy.interpolate import interp_tf
import trenfmodel


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
parser.add_argument('--burnin', type=int, default=100, help="number of iterations for burnin")
parser.add_argument('--tadapt', type=int, default=100, help="number of iterations for eps adaptation")
parser.add_argument('--ntrain', type=int, default=10, help="number of training iterations")
parser.add_argument('--thinning', type=int, default=20, help="thinning")
parser.add_argument('--lpsteps1', type=int, default=20, help="min leapfrog steps")
parser.add_argument('--lpsteps2', type=int, default=30, help="max leapfrog steps")
parser.add_argument('--mcmciter', type=int, default=200, help="number of only mcmc iterations")
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")



time.sleep(5)
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
nplot = 5
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
    lpsteps1, lpsteps2 = 2, 3
    nplot = 2

suffix = args.suffix
bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
donbody = False
order = args.order
shotnoise = bs**3/nc**3
dnoise = args.dnoise



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
params = tf.stack([cosmodata['Omega_c'], cosmodata['sigma8'], cosmodata['Omega_b'], cosmodata['h']])
params = tf.stack([cosmodata['Omega_c'], cosmodata['sigma8'], cosmodata['h']])
params = tf.stack([cosmodata['Omega_c'], cosmodata['sigma8']])
print("params : ", params)
ndim = params.numpy().size

if donbody: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc_gibbs/L%04d_N%04d_T%02dT_np%d'%(bs, nc, nstepsc, ndim)
elif order == 2: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc_gibbs/L%04d_N%04d_LPT_np%d'%(bs, nc, ndim)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc_gibbs/L%04d_N%04d_ZA_np%d'%(bs, nc, ndim)
elif order == 0: fpath = '/mnt/ceph/users/cmodi/galference/cosmo_hmc_gibbs/L%04d_N%04d_IC_np%d'%(bs, nc, ndim)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)

##################
##Generate DATA

np.random.seed(100)
zic = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
noise = np.random.normal(0, dnoise, nc**3).reshape(1, nc, nc, nc)
#ic = zic.copy()
#fin = zic.copy()
#data = zic.copy()
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
def whitenoise_to_linear(evolve, white, pk):
    nc, bs = evolve.nc, evolve.bs
    pkmesh = pk(evolve.kmesh)
    whitec = flowpm.utils.r2c3d(white* nc**1.5)
    lineark = tf.multiply(whitec, tf.cast((pkmesh /bs**3)**0.5, whitec.dtype))
    linear = flowpm.utils.c2r3d(lineark, norm=nc**3)
    return linear


@tf.function
def cosmo_sim(params, white, retic=False):
    if ndim == 3: cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1], h=params[2])
    if ndim == 2: cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1])
    #cosmology = flowpm.cosmology.Planck15(Omega_c=params[0], sigma8=params[1], h=params[2])
    k = tf.constant(np.logspace(-4, 1, 256), dtype=tf.float32)
    pk = flowpm.tfpower.linear_matter_power(cosmology, k)
    pk_fun = lambda x: tf.cast(
        tf.reshape(interp_tf(
                tf.reshape(tf.cast(x, tf.float32), [-1]), k, pk), x.shape), tf.
        complex64)
    ic = whitenoise_to_linear(evolve, white, pk_fun)
    if retic: return ic
    final_field = evolve.pm(ic, cosmology.to_dict())
    return final_field



##############################################
@tf.function
def unnormalized_log_prob_phases(white, params):
    final_field = cosmo_sim(params, white)
    loglik = -0.5 * tf.reduce_sum(((data- final_field)/dnoise)**2)
    #
    logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(white), axis=(-3, -2, -1)) 
    loss = loglik + logprior
    return loss

@tf.function
def grad_log_prob_phases(white, params):
    with tf.GradientTape() as tape:
        tape.watch(white)
        loss = unnormalized_log_prob_phases(white, params)
    return tape.gradient(loss, white)


@tf.function
def unnormalized_log_prob_params(params,  white):
    final_field = cosmo_sim(params, white)
    loglik = -0.5 * tf.reduce_sum(((data- final_field)/dnoise)**2)
    #
    logprior = tfd.Uniform(0.15, 0.35).log_prob(params[0]) 
    logprior += tfd.Uniform(0.65, 0.95).log_prob(params[1])
    if ndim == 3 : logprior += tfd.Uniform(0.55, 0.80).log_prob(params[2])
    loss = loglik + logprior
    return loss

@tf.function
def grad_log_prob_params(params,  white):
    with tf.GradientTape() as tape:
        tape.watch(params)
        loss = unnormalized_log_prob_params(params, white)
    return tape.gradient(loss, params)


py_log_prob_z = lambda x, y: unnormalized_log_prob_phases(tf.constant(x, dtype=tf.float32), 
                                                          tf.constant(y, dtype=tf.float32)).numpy().astype(np.float32)
py_grad_log_prob_z = lambda x, y: grad_log_prob_phases(tf.constant(x, dtype=tf.float32), 
                                                       tf.constant(y, dtype=tf.float32)).numpy().astype(np.float32)
hmckernel_z = PyHMC(py_log_prob_z, py_grad_log_prob_z, returnV=True)

py_log_prob_q = lambda x, y: unnormalized_log_prob_params(tf.constant(x, dtype=tf.float32), 
                                                          tf.constant(y, dtype=tf.float32)).numpy().astype(np.float32)
py_grad_log_prob_q = lambda x, y: grad_log_prob_params(tf.constant(x, dtype=tf.float32), 
                                                       tf.constant(y, dtype=tf.float32)).numpy().astype(np.float32)
hmckernel_q = PyHMC(py_log_prob_q, py_grad_log_prob_q, returnV=True)


################################################################################
####Setup

np.random.seed(2021 + device*100 + rank*3)
cosmodata = evolve.cosmodict
#params0 = tf.Variable((params.numpy()*np.random.uniform(0.9, 1.1)).astype(np.float32))
params0 = (params.numpy()*np.random.uniform(0.9, 1.1)).astype(np.float32)
zicmix = np.random.normal(0, 1, nc**3).reshape(1, nc, nc, nc).astype(np.float32)
white0 = 0.5*(zic + zicmix) #(zic*np.random.uniform(0.9, 1.1, zic.size).reshape(zic.shape)).astype(np.float32)
print(params, params0)

print("\nSetting up HMC\n")
start = time.time()
#######Correct log prob
stepsizez = np.float32(0.01)#args.eps
stepsizeq = np.float32(0.002) 
epsadaptz = DualAveragingStepSize(stepsizez)
epsadaptq = DualAveragingStepSize(stepsizeq)
samplesq, pyaccq = [], []
samplesz, pyaccz = [], []
psz, pss, pssx, pszx = [], [], [], []
q = params0.copy()
z = white0.copy()

print('Check z')
print("log prob z in device %d : "%device, py_log_prob_z(z, q))
#print("log prob z : ", py_grad_log_prob_z(z, q))

print('Check q')
print("log prob q in device %d : "%device, py_log_prob_q(q, z))
#print("log prob z : ", py_grad_log_prob_q(q, z))

###################################################################################




def qiteration(i, q, z, stepsize, warmup=False):
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    q, _, acc, energy, _ = hmckernel_q.hmc_step(q.copy(), lpsteps, stepsize, [z])
    prob = np.exp(energy[0] - energy[1])
    print('Accept/Reject in device %d for iteration %d of params: '%(device, i), acc, prob)

    if i < tadapt:
        if np.isnan(prob): prob = 0.
        if prob > 1: prob = 1.
        stepsize, avgstepsize = epsadaptq.update(prob)
        print('q stepsize', stepsize, avgstepsize)
    elif i == tadapt:
        _, stepsize = epsadaptq.update(prob)
        print("Step size fixed to : ", stepsize)
        np.save(fpath + '/stepsizesq%d-%02d'%(device, rank), stepsize)

    if warmup: return q, stepsize
    pyaccq.append(acc)
    samplesq.append(q.astype(np.float32)) 
    #append
    if (i%thinning) == 0:
        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
        print("Acceptance for parameters in device %d, rank %d = "%(device, rank), list(zip(np.unique(pyaccq), np.unique(pyaccq, return_counts=True)[1]/np.array(pyaccq).size)))
        np.save(fpath + '/samplesq%d-%02d'%(device, rank), np.array(samplesq))
        np.save(fpath + '/acceptsq%d-%02d'%(device, rank), np.array(pyaccq))

        npsamplesq = np.array(samplesq).astype(np.float32)
        fig = corner(npsamplesq, params.numpy())
        plt.savefig(fpath + '/figs/qcorner%02d-%05d'%(device, i))
        plt.close()
        fig, ax = plt.subplots(ndim, 1, figsize=(5, ndim*3), sharex=True)
        for j in range(ndim):
            ax[j].plot(npsamplesq[:, j])
        plt.savefig(fpath + '/figs/qiter%02d-%05d'%(device, i))
        plt.close()
    return q, stepsize


def ziteration(i, z, q, stepsize, warmup=False):
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    z, _, acc, energy, _ = hmckernel_z.hmc_step(z.copy(), lpsteps, stepsize, [q])
    prob = np.exp(energy[0] - energy[1])
    print('Accept/Reject in device %d for iteration %d of phases: '%(device, i), acc, prob)
    if i < tadapt:
        if np.isnan(prob): prob = 0.
        if prob > 1: prob = 1.
        stepsize, avgstepsize = epsadaptz.update(prob)
        print('z stepsize', stepsize, avgstepsize)
    elif i == tadapt:
        _, stepsize = epsadaptz.update(prob)
        print("Step size fixed to : ", stepsize)
        np.save(fpath + '/stepsizesz%d-%02d'%(device, rank), stepsize)

    if warmup : return z, stepsize
    pyaccz.append(acc)
    if (i%thinning) == 0: samplesz.append(z.astype(np.float32)) 

    #save power spectrum
    s = cosmo_sim(q, z, retic=True).numpy()
    pss.append(tools.power(s+1, boxsize=evolve.bs)[1])
    psz.append(tools.power(z+1, boxsize=evolve.bs)[1])
    pssx.append(tools.power(s+1, ic[0] + 1, boxsize=evolve.bs)[1])
    pszx.append(tools.power(z+1, zic[0] + 1, boxsize=evolve.bs)[1])

    #append
    if (i%thinning) == 0:
        print("Finished iteration %d on device %d, rank %d in %0.2f minutes"%(i, device, rank, (time.time()-start)/60.))
        print("Acceptance for phases in device %d, rank %d = "%(device, rank), list(zip(np.unique(pyaccz), np.unique(pyaccz, return_counts=True)[1]/np.array(pyaccz).size)))
        np.save(fpath + '/samplesz%d-%02d'%(device, rank), np.array(samplesz))
        np.save(fpath + '/acceptsz%d-%02d'%(device, rank), np.array(pyaccz))
        np.save(fpath + '/psz%d-%02d'%(device, rank), np.array(psz))
        np.save(fpath + '/pss%d-%02d'%(device, rank), np.array(pss))
        np.save(fpath + '/pssx%d-%02d'%(device, rank), np.array(pssx))
        np.save(fpath + '/pszx%d-%02d'%(device, rank), np.array(pszx))

        #q0 = np.array(samplesq).mean(axis=0).astype(np.float32)
        try:
            ics = [cosmo_sim(samplesq[j], samplesz[j], retic=True) for j in range(-nplot,0,1)]
            fig = callback_sampling(ics, ic, bs)
            plt.savefig(fpath + '/figs/siter%02d-%05d'%(device, i))
            plt.close()
        except Exception as e: print("######## Exception : ", e)
        try:
            fig = callback_sampling(tf.constant(samplesz[-nplot:]), zic, bs)
            plt.savefig(fpath + '/figs/ziter%02d-%05d'%(device, i))
            plt.close()
        except Exception as e: print("######## Exception : ", e)
    return z, stepsize

    

print("\nStart sampling\n")

for i in range(burnin):
    for _ in range(1): z, stepsizez = ziteration(i, z.copy(), q.copy(), stepsizez, warmup=True)
    for _ in range(1): q, stepsizeq = qiteration(i, q.copy(), z.copy(), stepsizeq, warmup=True)
 
for i in range(burnin, args.nsamples):
    z, stepsizez = ziteration(i, z.copy(), q.copy(), stepsizez)
    q, stepsizeq = qiteration(i, q.copy(), z.copy(), stepsizeq)



##########
#samples = np.array(samples)
#np.save(fpath + '/samples%d-%02d'%(device, rank), np.array(samples))
#np.save(fpath + '/accepts%d-%02d'%(device, rank), np.array(pyacc))
end = time.time()
print('Time taken in rank %d= '%rank, end-start)




