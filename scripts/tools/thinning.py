import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fpath', type=str, help='path')
parser.add_argument('--thin', type=int)
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
sys.path.append('../../galference/utils/')


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

#evolve = Evolve(nc, bs, a0=a0, af=af, nsteps = nsteps, donbody=donbody, order=order)     
ic = np.load(fpath + '/ic.npy', mmap_mode="r")

for i in range(4):
    jobid = i
    pss, psslin, pssx = [], [], []
    chunk = 100

    try: 
        allsamples = np.load(fpath + '/samples%d-00.npy'%i)[::args.thin]
        acc = np.load(fpath + '/accepts%d-00.npy'%i, mmap_mode="r")        
    except Exception as e:
        print(e)
        jobid = None
        #allsamples = np.load(fpath + '/samples.npy', mmap_mode="r+")
        allsamples = np.load(fpath + '/samples.npy', "r")[::args.thin] * 1.
        acc = np.load(fpath + '/accepts.npy', mmap_mode="r+")        

    print(allsamples.shape, acc.shape)
    print("zero entries : ", (allsamples == 0).sum())
    prethin = acc.shape[0]/allsamples.shape[0]
    print("Saved file is to thinned by ", prethin)
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

    if jobid is not None: np.save(fpath + '/samplesthin%d-%d-00.npy'%(args.thin, i), allsamples)
    else: np.save(fpath + '/samplesthin%d.npy'%(args.thin), allsamples)

    if jobid is None: break
