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
parser.add_argument('--nchains', type=int, default=1, help="number of chains")
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--reconiter', type=int, default=100, help="number of iterations for reconstruction")
parser.add_argument('--burnin', type=int, default=200, help="number of iterations for burnin")
parser.add_argument('--tadapt', type=int, default=50, help="number of iterations for eps adaptation")
parser.add_argument('--ntrain', type=int, default=10, help="number of training iterations")
parser.add_argument('--thinning', type=int, default=20, help="thinning")
parser.add_argument('--lpsteps1', type=int, default=25, help="min leapfrog steps")
parser.add_argument('--lpsteps2', type=int, default=50, help="max leapfrog steps")
parser.add_argument('--mcmciter', type=int, default=5, help="number of only mcmc iterations")
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")
parser.add_argument('--mode', type=str, default="classic", help='mode for trenf')
parser.add_argument('--nlayers', type=int, default=3, help='number of trenf layers')
parser.add_argument('--nbins', type=int, default=32, help='number of trenf layers')
parser.add_argument('--nknots', type=int, default=100, help='number of trenf layers')
parser.add_argument('--linknots', type=int, default=0, help='linear spacing for knots')
parser.add_argument('--kwts', type=int, default=4, help='number of trenf layers')
parser.add_argument('--prentrain', type=int, default=1000, help='number of iterations for training')
parser.add_argument('--probjump', type=float, default=0.3, help='probability of jump')
parser.add_argument('--preload', type=int, default=0, help='load already trained VI')

args = parser.parse_args()
device = args.jobid
suffix = args.suffix

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
#print("\nDevice name\n", tf.test.gpu_device_name(), "\n")
print("\nDevices\n", get_available_gpus())


import sys, os, time
import flowpm
sys.path.append('../../galference/utils/')
import tools
import diagnostics as dg

sys.path.append('../src/')
from pmfuncs import Evolve
from pyhmc import PyHMC, PyHMC_batch, DualAveragingStepSize
from trenfmodel import VItrenf
from hmcfuncs import DM_config, DM_fourier, Kwts
import recon, trenfmodel
from callback import callback, datafig, callback_fvi, callback_sampling

##########
bs, nc = args.bs, args.nc

#trenfpath = "/mnt/ceph/users/cmodi/galference/dm_fvi/L0200_N0064_ZA-burnin-affine/weights/iter0004"
#trenfpath = "/mnt/ceph/users/cmodi/galference/dm_fvi/L1000_N0128_ZA-affine/weights/iter0004"
samplepath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L1000_N0128_ZA-kwts4-fourier-corr/'
if nc == 32: samplepath = '//mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0032_ZA-check/'
else: samplepath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA-kwts4-fourier-corr/'%(bs, nc)

nchains = args.nchains
reconiter = args.reconiter
burnin = args.burnin
mcmciter = args.mcmciter
ntrain = args.ntrain
tadapt = args.tadapt
thinning = args.thinning
lpsteps1, lpsteps2 = args.lpsteps1, args.lpsteps2
prentrain = args.prentrain
if args.debug:
    reconiter = 5
    burnin = 5
    mcmciter = 5
    tadapt = 5
    thinning = 2 
    lpsteps1, lpsteps2 = 3, 5
    prentrain = 1

probjump = args.probjump
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
if donbody: fpath = '/mnt/ceph/users/cmodi/galference/dm_tsc/L%04d_N%04d_T%02d'%(bs, nc, nsteps)
elif order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_tsc/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_tsc/L%04d_N%04d_ZA'%(bs, nc)
if suffix == "": fpath = fpath + '/'
else: fpath = fpath + "-" + suffix + '/'
os.makedirs('%s'%fpath, exist_ok=True)
os.makedirs('%s'%fpath + '/figs/', exist_ok=True)
os.makedirs('%s'%fpath + '/opt/', exist_ok=True)
os.makedirs('%s'%fpath + '/weights/', exist_ok=True)
os.makedirs('%s'%fpath + '/burnin/', exist_ok=True)


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
k, pic = tools.power(ic[0], boxsize=bs)
k, pf = tools.power(fin[0], boxsize=bs)
k, pd = tools.power(data[0], boxsize=bs)
k, pn = tools.power(1+noise[0], boxsize=bs)
knoise = evolve.kmesh[evolve.kmesh > k[(pn > pf)][0]].min()
print("Noise dominated after : ", knoise, (evolve.kmesh > knoise).sum()/nc**3)

    

##############################################

@tf.function
def unnormalized_log_prob(zlatent):
    #z = tf.reshape(z, data.shape)
    #Chisq
    z = vitrenf.bijector.forward(zlatent)
    final_field = evolve.pmz(z)
    logprob = tfd.Normal(final_field, scale=dnoise).log_prob(tf.constant(data))
    logprob = tf.reduce_sum(logprob, axis=[-3, -2, -1])
    logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(z), axis=[-3, -2, -1])
    logjac = vitrenf.bijector.forward_log_det_jacobian(zlatent, event_ndims=3)
    print("z shape : ", z.shape)
    print("logprob : ", logprob)
    print("logprior : ", logprior)
    print("logjac : ", logjac)
    return  logprob + logprior + logjac


@tf.function
def grad_log_prob(z):
    with tf.GradientTape() as tape:
        tape.watch(z)
        logposterior = unnormalized_log_prob(z)
    grad = tape.gradient(logposterior, z)
    return grad


py_log_prob = lambda x: unnormalized_log_prob(tf.constant(x, dtype=tf.float32)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: grad_log_prob(tf.constant(x, dtype=tf.float32)).numpy().astype(np.float32)
hmckernel = PyHMC(py_log_prob, py_grad_log_prob, returnV=True)



#$#dmfuncs = DM_fourier(evolve, tfdata, dnoise=dnoise)
#$#kwts = None #Kwts(evolve, mode=args.kwts, knoise=knoise)
#$#py_log_prob = lambda x: unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
#$#py_grad_log_prob = lambda x: grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
hmckernel = PyHMC_batch(py_log_prob, py_grad_log_prob,  returnV=True)
stepsize = np.ones(nchains)*args.eps
epsadapt = DualAveragingStepSize(stepsize)
#stepsize = args.eps
samples, pyacc = [], []

print("HMC kernels setup")
##############################################


@tf.function
def fvi_grads(model, samples, opt):
    zsamples = samples
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        #q, logpq, acc = fvi_step(model, q, logpq)
        neglogq = - tf.reduce_mean(model.q.log_prob(zsamples))
    gradients = tape.gradient(neglogq, model.trainable_variables)
    return neglogq, gradients

def fvi_train(model, samples, opt):
    neglogq, gradients = fvi_grads(model, samples, opt)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return neglogq



def optimizetrenf(model, data, opt,  Rsm, niter=100, callback=callback, citer=50, nsamples=1, saveiter=20, sample=None, batch_size=int(8//nchains), stepsize=0.01):
    
    if sample is None: 
        sample = tf.random.normal([nsamples, nc, nc, nc])
    #logpsample = dmfuncs.unnormalized_log_prob(sample)
        
    accs, losses = [], []
    samples = []
    samples.append(sample)
    q = sample.copy()
    os.makedirs(fpath + '/opt', exist_ok=True)
    start = time.time()
    #dry run
    #sample, logpsample, acc, jump = fvi(model, sample, logpsample, 0, stepsize)
    #accs.append(acc)
    #samples.append(sample)
    for epoch in range(niter+1):
        print(epoch)

        lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
        q, _, acc, prob, _ = hmckernel.hmc_step(q, lpsteps, stepsize)
        lpq = tf.constant(prob[2]*-1.)
        prob = np.exp(prob[0] - prob[1])
        sample = tf.stop_gradient(model.bijector.forward(tf.constant(q, dtype=tf.float32)))
        #sample, logpsample, acc, jump = fvi(model, sample, logpsample, epoch, stepsize)
        accs.append(acc)
        if epoch % thinning == 0: samples.append(sample)
        ##Adapt stepsize
        print('Accept/Reject in device %d for iteratiion %d : '%(device, epoch), list(zip(acc, prob)))
        if epoch < tadapt:
            prob[np.isnan(prob)] = 0.
            prob[prob > 1] = 1.
            stepsize, avgstepsize = epsadapt.update(prob)
            print("stepsizes : ", ["%0.3f"%i for i in stepsize.flatten()])
            print("Avg stepsizes : ", ["%0.3f"%i for i in avgstepsize.flatten()])
        elif epoch == tadapt:
            _, stepsize = epsadapt.update(prob)
            print("Step size fixed to : ", stepsize)
            np.save(fpath + '/stepsizes%d-%02d'%(device, rank), stepsize)
            
        #Train network
        batch = sample #tf.concat([samples[i] for i in  list(idxsample)], axis=0)
        loss = fvi_train(model, batch, opt)
        losses.append(loss/nc**3)
        print("Logq at epoch %d is: "%epoch, loss.numpy())

        if epoch%citer == 0: 
            print("Time taken for %d iterations : "%citer, time.time() - start)
            print("Accept counts : ", np.unique(accs, return_counts=True))
            #
            fig = callback_fvi(model, ic, bs, losses)
            plt.savefig(fpath + '/figs/iter%05d'%epoch)
            plt.close()
            #
            print("Length samples : ", len(samples))
            fig = callback_sampling([evolve.z_to_lin(samples[-1])], ic, bs)
            #fig = callback_sampling([evolve.z_to_lin(i) for i in samples[-10:]], ic, bs)
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




def pretrain(vitrenf, train=True):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        1e-2,
        decay_steps=100,
        decay_rate=0.8,
        staircase=True)

    #Train FLOW using previos samples assuming we have them
    print("Load old samples")
    allsamples = []
    perchain = 200
    for i in range(4):
        #allsamples.append(np.load('/mnt/ceph/users/cmodi/galference/dm_hmc/L1000_N0128_ZA-kwts4-fourier/samples%d-00.npy'%i)[-200:, 0])
        f = np.load(samplepath + '/samples%d-00.npy'%i)[:perchain, 0]
        allsamples.append(f)
    allsamples = np.stack(allsamples, axis=1).astype(np.float32)
    print(allsamples.shape)
    if allsamples.shape[-1] ==2:
        allsamples2 = [evolve.zk_to_z(i) for i in allsamples]
        allsamples = np.array(allsamples2)
        print("After reshape : ", allsamples.shape)

    if train:
        nsamples = 2
        losses = []

        opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)
        saveiter = 100

        print("Train on old samples")
        for i in range(prentrain+1):
            batch = tf.concat([allsamples[idx] for idx in np.random.choice(allsamples.shape[0], nsamples)], axis=0)
            losses.append(fvi_train(vitrenf, batch, opt))
            if i%100 == 0: 
                print(i)
                fig = callback_fvi(vitrenf, ic, bs, losses)
                plt.savefig(fpath + 'figs/trainr%05d'%i)
                plt.close()
            if (i > 0) & (i%saveiter == 0): 
                vitrenf.save_weights(fpath + '/weights/preiter%04d'%(i//saveiter))

        print("Trained")
    return allsamples
    


################################################################
###################VI
print("\nStart VI\n")
vitrenf = trenfmodel.VItrenf(nc, nlayers=args.nlayers, evolve=evolve, nbins=args.nbins, nknots=args.nknots, mode=args.mode, linknots=bool(args.linknots))
for i in vitrenf.variables:
    print(i.name, i.shape)

if args.preload:
    try:
        print("Loading pre-trained newtork from %s"%(fpath + '/weights/preiter%04d'%(args.prentrain//100)))
        vitrenf.load_weights(fpath + '/weights/preiter%04d'%(args.prentrain//100))
        allsamples = pretrain(vitrenf, train=False)
    except Exception as e:
        print(e)
        allsamples = pretrain(vitrenf)
else:
        allsamples = pretrain(vitrenf)


#$#allsamples = []
#$#for i in range(args.nchains):
#$#    f = np.load(samplepath + '/samples%d-00.npy'%i)[:, 0]
#$#    allsamples.append(f)
#$#
#$#allsamples = np.stack(allsamples, axis=1).astype(np.float32)
#$##stepsize = np.array([np.load(samplepath + '/stepsizes%d-00.npy'%i) for i in range(args.nchains)]).flatten()
stepsize = np.array([args.eps for i in range(args.nchains)]).flatten()

sample = tf.constant(allsamples[-1])
sample = vitrenf.bijector.inverse(sample).numpy()
print(sample.shape)
print("stepsize : ", stepsize)
#del allsamples

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    5e-3,
    decay_steps=100,
    decay_rate=0.8,
    staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)
R = 0.
losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 5001, callback, citer=5, sample=sample, stepsize=stepsize)

