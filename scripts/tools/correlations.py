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
parser.add_argument('--fpath', type=str, help='path')
parser.add_argument('--thin', type=int, default=1)
parser.add_argument('--jobid', type=int, default=0, help='an integer for the accumulator')
parser.add_argument('--eps', type=float, default=0.001, help='step size')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--nc', type=int, default=32, help='mesh size')
parser.add_argument('--bs', type=float, default=200, help='box size')
parser.add_argument('--order', type=int, default=1, help='ZA/LPT ordr')
parser.add_argument('--suffix', type=str, default="", help='suffix to fpath')

args = parser.parse_args()
device = args.jobid
suffix = args.suffix


import sys, os, time
import flowpm
import arviz as az
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


bs, nc = args.bs, args.nc
nsteps = 3
a0, af, nsteps = 0.1, 1.0,  nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
nsims = 200
donbody = False
order = args.order
shotnoise = bs**3/nc**3
dnoise = 1. #shotnoise/nc**1.5  

fpath = args.fpath
print(fpath)

evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     
ic = np.load(fpath + '/ic.npy', mmap_mode="r")

#$#
#$#def saveps(allsamples, ic, jobid, fourier, drop):
#$#    ssz, sslin = [], []
#$#    pss, psslin = [], []
#$#    pssx = []
#$#
#$#    for i in range(allsamples.shape[0]):
#$#        if fourier:
#$#                ssz = evolve.zk_to_z(allsamples[i]).numpy()
#$#        else: ssz = allsamples[i].copy()
#$#        sslin = evolve.z_to_lin(tf.constant(ssz)).numpy()
#$#        #
#$#        pss.append([tools.power(j+1, boxsize=evolve.bs)[1] for j in ssz])
#$#        psslin.append([tools.power(j+1, boxsize=evolve.bs)[1] for j in sslin])
#$#        pssx.append([tools.power(j+1, ic[0] + 1, boxsize=evolve.bs)[1] for j in sslin])
#$#        if i%100 == 0 :
#$#            print('iteration ', i, ssz.shape, ssz.mean(), sslin.mean())
#$#    pss = np.stack(pss)
#$#    psslin = np.stack(psslin)
#$#    pssx = np.stack(pssx)
#$#    print(pss.shape, psslin.shape)
#$#    if drop == 1: drop = ""
#$#    if jobid is not None:
#$#        np.save(fpath + '/psz%s-%d'%jobid, pss)
#$#        np.save(fpath + '/pslin%s-%d'%jobid, psslin) #
#$#        np.save(fpath + '/pslinx%s-%d'%jobid, pssx) #
#$#    else:
#$#        np.save(fpath + '/psz%s'%drop, pss)
#$#        np.save(fpath + '/pslin%s'%drop, psslin) #
#$#        np.save(fpath + '/pslinx%s'%drop, pssx) #
#$#
#$#    rcc, rcclin, rcl = [], [], []
#$#    for j in range(1, pss.shape[2]):
#$#        acc = [az.autocorr(pss[:, ic, j]) for ic in range(pss.shape[1])]
#$#        acclin = [az.autocorr(psslin[:, ic, j]) for ic in range(pss.shape[1])]
#$#        acl = [np.where(acc[ic] < 0.1)[0][0] * thin for ic in range(pss.shape[1])]
#$#        aclinl = [np.where(acclin[ic] < 0.1)[0][0] * thin for ic in range(pss.shape[1])]
#$#        #print(j, aclinl)
#$#        rcc.append(acc)
#$#        rcclin.append(acclin)
#$#        rcl.append([acl, aclinl])
#$#
#$#    rcc, rcclin, rcl = np.array(rcc), np.array(rcclin), np.array(rcl)
#$#    print(rcc.shape, rcclin.shape, rcl.shape)
#$#    if jobid is not None:
#$#        np.save(fpath + '/acc%d'%jobid, rcc)
#$#        np.save(fpath + '/acclin%d'%jobid, rcclin)
#$#        np.save(fpath + '/aclength%d'%jobid, rcl)
#$#    else:
#$#        np.save(fpath + '/acc', rcc)
#$#        np.save(fpath + '/acclin', rcclin)
#$#        np.save(fpath + '/aclength', rcl)
#$#
#$#

for i in range(4):
    jobid = i
    pss, psslin, pssx = [], [], []
    chunk = 100
    print("args thin : ", args.thin)

    try: 
        if args.thin == 1: allsamples = np.load(fpath + '/samples%d-00.npy'%i, mmap_mode="r")[::, 0]
        else: allsamples = np.load(fpath + '/samplesthin%d-%d-00.npy'%(args.thin, i), mmap_mode="r")
        acc = np.load(fpath + '/accepts%d-00.npy'%i, mmap_mode="r")        
    except Exception as e:
        print(e)
        jobid = None
        if args.thin == 1 :
            allsamples = np.load(fpath + '/samples.npy', mmap_mode="r+")
        else: allsamples = np.load(fpath + '/samplesthin%d.npy'%(args.thin))
        acc = np.load(fpath + '/accepts.npy', mmap_mode="r+")        

    print(allsamples.shape, acc.shape)
    thin = acc.shape[0]/allsamples.shape[0]
    print("Thinning by ", thin)
    if allsamples.shape[-1] == 2: 
        fourier = True
        if len(allsamples.shape) == 5 : 
            allsamples = np.expand_dims(allsamples, 1)
    else : 
        fourier = False
        if len(allsamples.shape) == 4 :
            allsamples = np.expand_dims(allsamples, 1)    
    print("reshape " , allsamples.shape)
    print("Fourier ", fourier)
    #

    
    
    for i in range(allsamples.shape[0]):
        if fourier:
            ssz = evolve.zk_to_z(allsamples[i]).numpy()
        else: ssz = allsamples[i].copy()
        sslin = evolve.z_to_lin(tf.constant(ssz)).numpy()
        #
        pss.append([tools.power(j+1, boxsize=evolve.bs)[1] for j in ssz])
        psslin.append([tools.power(j+1, boxsize=evolve.bs)[1] for j in sslin])
        pssx.append([tools.power(j+1, ic[0] + 1, boxsize=evolve.bs)[1] for j in sslin])
        if i%99 == 0 : print('iteration ', i, ssz.mean(), sslin.mean())
    del allsamples

    pss = np.stack(pss)
    psslin = np.stack(psslin)
    pssx = np.stack(pssx)
    print(pss.shape, psslin.shape)
    suffix = ""
    if args.thin != 1: suffix = suffix + 'thin%d'%(args.thin)
    if jobid is not None: suffix = suffix + '-%d'%jobid
    np.save(fpath + '/psz%s'%suffix, pss)
    np.save(fpath + '/pslin%s'%suffix, psslin) #
    np.save(fpath + '/pslinx%s'%suffix, pssx) #

    rcc, rcclin, rcl = [], [], []
    for j in range(1, pss.shape[2]):
        acc = [az.autocorr(pss[:, ic, j]) for ic in range(pss.shape[1])]
        acclin = [az.autocorr(psslin[:, ic, j]) for ic in range(pss.shape[1])]
        acl = [np.where(acc[ic] < 0.1)[0][0] * thin for ic in range(pss.shape[1])]
        aclinl = [np.where(acclin[ic] < 0.1)[0][0] * thin for ic in range(pss.shape[1])]
        #print(j, aclinl)
        rcc.append(acc)
        rcclin.append(acclin)
        rcl.append([acl, aclinl])

    rcc, rcclin, rcl = np.array(rcc), np.array(rcclin), np.array(rcl)
    print(rcc.shape, rcclin.shape, rcl.shape)
    np.save(fpath + '/acc%s'%suffix, rcc)
    np.save(fpath + '/acclin%s'%suffix, rcclin)
    np.save(fpath + '/aclength%s'%suffix, rcl)
    #
    if jobid is None : break




#$#    for ichunk in range(nchunks):
#$#        i0, i1 = ichunk*chunk, ichunk*chunk + chunk
#$#        if jobid is not None: 
#$#            allsamples = np.load(fpath + '/samples%d-00.npy'%i, mmap_mode="r")[i0:i1]
#$#        else: 
#$#            allsamples = np.load(fpath + '/samples.npy')[i0:i1]
#$#
#$#        print(ichunk, i0, i1, allsamples[0].std(), allsamples[-1].std())
#$#        for i in range(allsamples.shape[0]):
#$#            if fourier:
#$#                    ssz = evolve.zk_to_z(allsamples[i]).numpy()
#$#            else: ssz = allsamples[i].copy()
#$#            sslin = evolve.z_to_lin(tf.constant(ssz)).numpy()
#$#            #
#$#            pss.append([tools.power(j+1, boxsize=evolve.bs)[1] for j in ssz])
#$#            psslin.append([tools.power(j+1, boxsize=evolve.bs)[1] for j in sslin])
#$#            pssx.append([tools.power(j+1, ic[0] + 1, boxsize=evolve.bs)[1] for j in sslin])
#$#            if i%99 == 0 : print('iteration ', i, ssz.mean(), sslin.mean())
#$#        del allsamples
