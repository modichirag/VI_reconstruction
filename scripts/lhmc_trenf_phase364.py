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
parser.add_argument('--ntrain', type=int, default=100, help="number of training iterations")
parser.add_argument('--prentrain', type=int, default=1000, help="number of training iterations")
parser.add_argument('--thinning', type=int, default=20, help="thinning")
parser.add_argument('--kwts', type=int, default=4, help='number of trenf layers')
parser.add_argument('--lpsteps1', type=int, default=25, help="min leapfrog steps")
parser.add_argument('--lpsteps2', type=int, default=50, help="max leapfrog steps")
parser.add_argument('--mcmciter', type=int, default=5, help="number of only mcmc iterations")
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")
#
parser.add_argument('--mode', type=str, default="classic", help='mode for trenf')
parser.add_argument('--nlayers', type=int, default=3, help='number of trenf layers')
parser.add_argument('--nbins', type=int, default=32, help="number of bins in trenf spline")
parser.add_argument('--fitnoise', type=int, default=1, help='fitnoise')
parser.add_argument('--fitscale', type=int, default=1, help='fitscale')
parser.add_argument('--fitmean', type=int, default=1, help='fitmean')
parser.add_argument('--meanfield', type=int, default=1, help='meanfield for affine')
parser.add_argument('--nknots', type=int, default=100, help='number of trenf layers')
parser.add_argument('--linknots', type=int, default=0, help='linear spacing for knots')
#
parser.add_argument('--probjump', type=float, default=0.3, help='probability of jump')
parser.add_argument('--regwt0', type=float, default=0., help='regularization weight')
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
samplepath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L%04d_N%04d_ZA-kwts4-fourier-corr/'%(args.bs, args.nc)
#samplepath = '/mnt/ceph/users/cmodi/galference/dm_hmc/L0200_N0064_ZA-fourier-minlininit2/'
nchains = args.nchains
reconiter = args.reconiter
burnin = args.burnin
mcmciter = args.mcmciter
ntrain = args.ntrain
tadapt = args.tadapt
thinning = args.thinning
lpsteps1, lpsteps2 = args.lpsteps1, args.lpsteps2
nsamples = 4
prentrain = args.prentrain
nR = args.nR
regwt0 = args.regwt0
checkprobs = 10 
probjump = args.probjump
saveiter = 100
#
if args.debug:
    reconiter = 5
    burnin = 5
    mcmciter = 5
    tadapt = 5
    thinning = 2 
    lpsteps1, lpsteps2 = 3, 5
    nsamples = 1
    prentrain = 20
    nR = 0 
    checkprobs = 2
allRs = [0., 1., 2., 4.]
allR = allRs[:nR + 1][::-1]

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
if donbody: fpath = '/mnt/ceph/users/cmodi/galference/dm_lhmc/L%04d_N%04d_T%02d'%(bs, nc, nsteps)
elif order == 2: fpath = '/mnt/ceph/users/cmodi/galference/dm_lhmc/L%04d_N%04d_LPT'%(bs, nc)
elif order == 1: fpath = '/mnt/ceph/users/cmodi/galference/dm_lhmc/L%04d_N%04d_ZA'%(bs, nc)
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

    
dmfuncs = DM_fourier(evolve, tfdata, dnoise=dnoise)
kwts = Kwts(evolve, mode=args.kwts, knoise=knoise)
py_log_prob = lambda x: dmfuncs.unnormalized_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
py_grad_log_prob = lambda x: dmfuncs.grad_log_prob(tf.constant(x, dtype=tf.float32), tf.constant(1.)).numpy().astype(np.float32)
hmckernel = PyHMC_batch(py_log_prob, py_grad_log_prob, invmetric_diag=kwts.kwts, returnV=True)
epsadapt = DualAveragingStepSize(args.eps)
stepsize = args.eps
samples, pyacc = [], []

###VI
vitrenf = trenfmodel.VItrenf(nc, nlayers=args.nlayers, evolve=evolve, nbins=args.nbins, nknots=args.nknots, mode=args.mode, \
                             linknots=bool(args.linknots), fitnoise=bool(args.fitnoise), fitscale=bool(args.fitscale), \
                             fitmean=bool(args.fitmean), meanfield=bool(args.meanfield))
for i in vitrenf.variables:
    print(i.name, i.shape)

#####                                                                                                                                                                                                                                                                                                                       
@tf.function
def unnormalized_log_prob_latent(z):
    return vitrenf.noise.log_prob(z)

@tf.function
def grad_log_prob_latent(z):
    with tf.GradientTape() as tape:
        tape.watch(z)
        logprob = unnormalized_log_prob_latent( z)       
    return tape.gradient(logprob, z)

py_log_prob_latent = lambda x: unnormalized_log_prob_latent(tf.constant(x, dtype=tf.float32)).numpy().astype(np.float32)
py_grad_log_prob_latent = lambda x: grad_log_prob_latent(tf.constant(x, dtype=tf.float32)).numpy().astype(np.float32)
hmckernel_latent = PyHMC_batch(py_log_prob_latent, py_grad_log_prob_latent)
##############################################




def flow_step(model, x, lpx):
    nsamples = x.shape[0]
    xk = tf.identity(x)
    x = evolve.zk_to_z(x)
    lqx = model.q.log_prob(x)
    
    #y = model.q.sample(nsamples)*1.
    q = vitrenf.bijector.inverse(x).numpy()
    for j in range(1):
        lpsteps = np.random.randint(25, 50, 1)[0]
        q = hmckernel_latent.hmc_step(q, lpsteps, np.array([0.01]))[0]
    y = vitrenf.bijector.forward(tf.constant(q, dtype=tf.float32))

    yk = evolve.z_to_zk(y)
    lpy = dmfuncs.unnormalized_log_prob(yk)
    lqy = model.q.log_prob(y)
    print(tf.stack([lpx, lpy, lqx, lqy], axis=0).numpy())
    prob = tf.exp((lpy - lpx)/nc**3 + lqx - lqy)
    #prob = tf.exp((lpy - lpx + lqx - lqy))

    accept = tf.where(tf.random.uniform([1]) <= tf.minimum(1., prob))
    reject = tf.where(tf.random.uniform([1]) > tf.minimum(1., prob))
    z = tf.scatter_nd(accept, tf.gather_nd(yk, accept), yk.shape) + \
         tf.scatter_nd(reject, tf.gather_nd(xk, reject), yk.shape)

    lpz = tf.where(tf.random.uniform([1]) <= tf.minimum(1., prob), lpy, lpx)
    acc = tf.where(tf.random.uniform([1]) <= tf.minimum(1., prob), tf.ones(lpx.shape)*11, tf.ones(lpx.shape)*10.)
    return z, lpz, prob, acc, tf.constant(1.)


def mcmc_step(q, stepsize):
    #stepsize = np.random.uniform(0.01, 0.02, 1)
    lpsteps = np.random.randint(lpsteps1, lpsteps2, 1)[0]
    print("mcmc : ", q.shape)
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
            for _ in range(5):
                print("Jump, recaliberate")
                q, logpq, prob2, acc2, jump2 =  mcmc_step(q, stepsize)
                
            
    print("prob ", prob.numpy(), "to accept/reject with ", acc.numpy(), " in time %0.3f"%(time.time() - start))
    return q, logpq, acc, jump


@tf.function
def fvi_grads(model, samples, opt, samples2, regwt):
    samples = evolve.zk_to_z(samples)    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        #q, logpq, acc = fvi_step(model, q, logpq)
        neglogq = - tf.reduce_mean(model.q.log_prob(samples))
        if regwt0 ==  0:
            loss = neglogq*1.
        else:
            logq2 = - tf.reduce_mean(model.q.log_prob(evolve.zk_to_z(samples2)))
            samplevi = model.sample(samples.shape[0])*1.
            logqvi = - tf.reduce_mean(model.q.log_prob(samplevi))
            reg = regwt * tf.reduce_mean((logq2 - logqvi)**2.)
            loss = neglogq + reg
            
    gradients = tape.gradient(loss, model.trainable_variables)
    return neglogq, gradients



def fvi_train(model, samples, opt, batch2, regwt):
    neglogq, gradients = fvi_grads(model, samples, opt, batch2, regwt)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return neglogq




def optimizetrenf(model, data, opt,  Rsm, niter=100, callback=callback, citer=50, nsamples=1, saveiter=20, sample=None, batch_size=int(16//nchains), stepsize=0.01, samples=[]):
    
    if sample is None: 
        sample = tf.random.normal([nsamples, nc, nc, nc])
    logpsample = dmfuncs.unnormalized_log_prob(sample)
        
    accs, losses = [], []
    samples.append(sample)
    idxtrain = list(np.arange(len(samples)))
    #
    start = time.time()
    #dry run
    sample, logpsample, acc, jump = fvi(model, sample, logpsample, 0, stepsize)
    print(sample.shape, samples[0].shape, samples[-1].shape)
    accs.append(acc)
    samples.append(sample)
    idxtrain = idxtrain + [len(samples)-1]
    trainsize = len(samples)
    print("Training size of %d is maintained"%trainsize)
    for epoch in range(niter+1):
        print(epoch)
        sample, logpsample, acc, jump = fvi(model, sample, logpsample, epoch, stepsize)
        accs.append(acc)
        samples.append(sample)
        idxtrain = idxtrain + [len(samples)-1]
        print(len(idxtrain))
        #Train the networks
        for itrain in range(2):
            if itrain > len(samples): break
            if jump == 1: 
                print("Jump, no train")
                break
            if itrain == 0: idxsample = idxtrain[-batch_size:]
            else: idxsample = np.random.choice(idxtrain, batch_size, replace=False)
            idxsample = list(idxsample)
            print("idx :", idxsample)
            batch = tf.concat([samples[i] for i in list(idxsample)], axis=0)
            batch2, regwt = None, tf.constant(regwt0)
            if regwt0 !=None:
                if itrain == 0: idxsample = idxtrain[-batch_size:]
                else: idxsample = np.random.choice(idxtrain, batch_size, replace=False)
                idxsample = list(idxsample)
                batch2 = tf.concat([samples[i] for i in list(idxsample)], axis=0)
            loss = fvi_train(model, batch, opt, batch2, regwt)
            losses.append(loss/nc**3)
            print("Logq at epoch %d is: "%epoch, loss.numpy())
        iddel = np.random.randint(trainsize)
        print("Remove index from train : %d "%idxtrain[iddel])
        del idxtrain[iddel]

        if epoch%citer == 0: 
            print("Time taken for %d iterations : "%citer, time.time() - start)
            print("Accept counts : ", np.unique(accs, return_counts=True))
            #
            fig = callback_fvi(model, ic, bs, losses)
            plt.savefig(fpath + '/figs/iter%05d'%epoch)
            plt.close()
            #
#$#            lqs = []
#$#            for j in range(checkprobs):
#$#                idxsample = np.random.choice(idxtrain, batch_size, replace=False)
#$#                idxsample = list(idxsample)
#$#                try:
#$#                    print(idxsample)
#$#                    batch = tf.concat([samples[i:i+2] for i in list(idxsample)], axis=0)
#$#                    batch = evolve.zk_to_z(batch)
#$#                    lq = model.q.log_prob(batch).numpy()
#$#                    plt.plot(lq.flatten(), 'C0.')
#$#                    lqs = lqs + list(lq.flatten())
#$#                    #plt.plot(lqs, '.', label='train samples', lw=2)
#$#                except Exception as e:
#$#                    print(e)
#$#            lqs = []
#$#            for j in range(checkprobs):
#$#                try:
#$#                    batch = model.sample(nchains)
#$#                    lq = model.q.log_prob(batch).numpy()
#$#                    lqs = lqs + list(lq.flatten())
#$#                    plt.plot(lq.flatten(), 'C1.', lw=2)
#$#                except Exception as e:
#$#                    print(e)
#$#            plt.savefig(fpath + '/figs/lgoqiter%05d'%epoch)
#$#            plt.close()
#$#

            #
            fig = callback_sampling([evolve.zk_to_lin(i) for i in samples[-2:]], ic, bs)
            plt.savefig(fpath + '/figs/samples%05d'%epoch)
            plt.close()
            np.save(fpath + 'loss', np.array(losses))
            np.save(fpath + 'samples', np.array(samples))
            np.save(fpath + 'accepts', np.array(accs))
            start = time.time()
        if epoch%saveiter == 0: 
            model.save_weights(fpath + '/weights/iter%04d'%(epoch//saveiter))
            #np.save(fpath + '/opt/iter%04d'%(epoch//saveiter), opt.get_weights(), allow_pickle=True)
            #print('Weights saved')
            
    return losses 




#$####################################################################################
#$################################################################
###################VI



def pretrain(vitrenf, train=True):

    print("Load old samples")
    allsamples = []
    perchain = 499
    for i in range(nchains):
        f = np.load(samplepath + '/samples%d-00.npy'%i)[:perchain, 0]
        allsamples.append(f)
    allsamples = np.stack(allsamples, axis=1).astype(np.float32)
    print(allsamples.shape)

    losses = []

    if train:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-2,
            decay_steps=100,
            decay_rate=0.8,
            staircase=True)

        opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)

        print("Train on old samples")
        for i in range(prentrain):
            batch = tf.concat(allsamples[np.random.choice(allsamples.shape[0], nsamples)], axis=0)
            batch2, regwt = None, None
            if regwt0 !=0 :
                batch2 = tf.concat(allsamples[np.random.choice(allsamples.shape[0], nsamples)], axis=0)
                regwt = tf.constant(args.regwt0*i/prentrain)
            losses.append(fvi_train(vitrenf, batch, opt, batch2, regwt))
            if i%100 == 0: 
                fig = callback_fvi(vitrenf, ic, bs, losses)
                plt.savefig(fpath + 'figs/trainr%05d'%i)
                plt.close()

                lqs = []
                for j in range(allsamples.shape[0]//10):
                    batch = tf.concat(evolve.zk_to_z(allsamples[j:j+2]), axis=0)
                    lq = vitrenf.q.log_prob(batch).numpy()
                    lqs = lqs + list(lq.flatten())
                plt.plot(lqs, '.', label='train samples', lw=2)

                lqs = []
                for j in range(allsamples.shape[0]//10):
                    batch = vitrenf.sample(nchains)
                    lq = vitrenf.q.log_prob(batch).numpy()
                    lqs = lqs + list(lq.flatten())
                plt.plot(lqs, '.', label='flow samples', lw=2)
                plt.legend()
                plt.savefig(fpath + '/figs/lgoqtrain%05d'%i)
                plt.close()

            if (i > 0) & (i%saveiter == 0): 
                vitrenf.save_weights(fpath + '/weights/preiter%04d'%(i//saveiter))

    return allsamples



#####################################################
#Train FLOW using previos samples assuming we have them
if args.preload:
    try:
        print("Loading pre-trained newtork from %s"%(fpath + '/weights/preiter%04d'%(args.prentrain//saveiter -1)))
        vitrenf.load_weights(fpath + '/weights/preiter%04d'%(args.prentrain//saveiter - 1))
        allsamples = pretrain(vitrenf, train=False)
    except Exception as e:
        print(e)
        allsamples = pretrain(vitrenf)
else:
        allsamples = pretrain(vitrenf)



plt.imshow(vitrenf.sample(1)[0].numpy().sum(axis=0))
plt.colorbar()
plt.savefig(fpath + 'initsample')
plt.colorbar()
plt.close()

sample = tf.constant(allsamples[-1])
print(sample.shape)
R = 0.
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            5e-3,
            decay_steps=100,
            decay_rate=0.8,
            staircase=True)

opt = tf.keras.optimizers.Adam(learning_rate= lr_schedule)
stepsize = np.array([np.load(samplepath + '/stepsizes%d-00.npy'%i) for i in range(nchains)]).flatten()
print("stepsize : ", stepsize)
losses = optimizetrenf(vitrenf, tf.constant(data), opt, tf.constant(R), 5001, callback, citer=5, sample=sample, stepsize=stepsize, samples=list(allsamples))

