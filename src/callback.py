import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append('../../galference/utils/')
import tools
import diagnostics as dg

def callback(model, ic, bs, losses=None):
    
    fig, ax = plt.subplots(1, 6, figsize=(15, 3))
    im = ax[0].imshow(ic[0].sum(axis=0))
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title('Truth')
    #
    sample = model.sample_linear
    im = ax[1].imshow((sample).numpy()[0].sum(axis=0))
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title('Sample')
    #
    diff = sample - ic
    im = ax[2].imshow(diff.numpy()[0].sum(axis=0))
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title('Differnce')


    #2pt functions
    k, p0 = tools.power(ic[0]+1, boxsize=bs)
    ps, rc, ratios = [], [], []
    for i in range(20):
        sample = model.sample_linear
        i0 = (sample).numpy()[0]
        k, p1 = tools.power(i0+1, boxsize=bs)
        k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
        ps.append([p1, p1x])
        rc.append(p1x/(p1*p0)**0.5)
        ratios.append((p1/p0)**0.5)
    rc = np.array(rc)
    ratios = np.array(ratios)
    
    ax = ax[3:]
    ax[0].plot(k, rc.T, 'C1', alpha=0.2)
    ax[0].plot(k, rc.mean(axis=0))
    ax[0].semilogx()
    ax[0].set_ylim(0., 1.05)
    ax[0].set_title('$r_c$', fontsize=12)
    
    ax[1].plot(k, ratios.T, 'C1', alpha=0.2)
    ax[1].plot(k, ratios.mean(axis=0))
    ax[1].semilogx()
    ax[1].set_ylim(0.8, 1.2)
    ax[1].set_title('$t_f$', fontsize=12)
    
#     if losses is not None: ax[2].plot(losses)
    if losses is not None: 
        losses = -1. * np.array(losses)
        ax[2].plot(losses[:, 0], label='-logl')
        ax[2].plot(losses[:, 1], label='-logp')
        ax[2].plot(losses[:, 2], label='-logq')
        ax[2].plot(losses[:, 3], 'k', label='-elbo')
    ax[2].loglog()
    ax[2].set_title('-ELBO', fontsize=12)
    ax[2].legend()
    for axis in ax: axis.grid(which='both')
    
    plt.tight_layout()
    return fig




def callback_fvi(model, ic, bs, losses=None, zoomin=True):
    
    fig, ax = plt.subplots(1, 6, figsize=(15, 3))
    im = ax[0].imshow(ic[0].sum(axis=0))
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title('Truth')
    #
    sample = model.sample_linear
    im = ax[1].imshow((sample).numpy()[0].sum(axis=0))
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title('Sample')
    #
    diff = sample - ic
    im = ax[2].imshow(diff.numpy()[0].sum(axis=0))
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title('Differnce')


    #2pt functions
    k, p0 = tools.power(ic[0]+1, boxsize=bs)
    ps, rc, ratios = [], [], []
    for i in range(20):
        sample = model.sample_linear
        i0 = (sample).numpy()[0]
        k, p1 = tools.power(i0+1, boxsize=bs)
        k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
        ps.append([p1, p1x])
        rc.append(p1x/(p1*p0)**0.5)
        ratios.append((p1/p0)**0.5)
    rc = np.array(rc)
    ratios = np.array(ratios)
    
    ax = ax[3:]
    ax[0].plot(k, rc.T, 'C1', alpha=0.2)
    ax[0].plot(k, rc.mean(axis=0))
    ax[0].semilogx()
    ax[0].set_ylim(0., 1.05)
    ax[0].set_title('$r_c$', fontsize=12)
    
    ax[1].plot(k, ratios.T, 'C1', alpha=0.2)
    ax[1].plot(k, ratios.mean(axis=0))
    ax[1].semilogx()
    if zoomin: ax[1].set_ylim(0.8, 1.2)
    else: ax[1].set_ylim(0.0, 1.5)
    ax[1].set_title('$t_f$', fontsize=12)
    
    ax[2].plot(losses)
    ax[2].loglog()
    ax[2].set_title('-logq', fontsize=12)
    ax[2].legend()
    for axis in ax: axis.grid(which='both')
    
    plt.tight_layout()
    return fig




def callback_sampling(samples, ic, bs):
    
    fig, axar = plt.subplots(2, 3, figsize=(12, 8))
    ax = axar[0]
    im = ax[0].imshow(ic[0].sum(axis=0))
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title('Truth')
    #
    sample = samples[np.random.randint(len(samples))].numpy()
    im = ax[1].imshow((sample)[0].sum(axis=0))
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title('Sample')
    #
    diff = sample - ic
    print(diff.shape)
    im = ax[2].imshow(diff[0].sum(axis=0))
    plt.colorbar(im, ax=ax[2])
    ax[2].set_title('Differnce')


    #2pt functions
    k, p0 = tools.power(ic[0]+1, boxsize=bs)
    ps, rc, ratios = [], [], []
    for i in range(len(samples)):
        sample = samples[i].numpy()
        if len(sample.shape) == 4: 
            for j in range(sample.shape[0]):
                i0 = (sample)[j]
                k, p1 = tools.power(i0+1, boxsize=bs)
                k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
                ps.append([p1, p1x])
                rc.append(p1x/(p1*p0)**0.5)
                ratios.append((p1/p0)**0.5)
        elif len(sample.shape) == 3:
            i0 = sample.copy()
            k, p1 = tools.power(i0+1, boxsize=bs)
            k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
            ps.append([p1, p1x])
            rc.append(p1x/(p1*p0)**0.5)
            ratios.append((p1/p0)**0.5)
    rc = np.array(rc)
    ratios = np.array(ratios)
    
    ax = axar[1]
    ax[0].plot(k, rc.T, alpha=0.3)
    ax[0].plot(k, rc.mean(axis=0))
    ax[0].semilogx()
    ax[0].set_ylim(0., 1.05)
    ax[0].set_title('$r_c$', fontsize=12)
    
    ax[1].plot(k, ratios.T, alpha=0.3)
    ax[1].plot(k, ratios.mean(axis=0))
    ax[1].semilogx()
    ax[1].set_ylim(0.8, 1.2)
    ax[1].set_title('$t_f$', fontsize=12)
        
    ax[2].plot(k, p0, 'k', alpha=0.8)
    for ip in ps:
        ax[2].plot(k, ip[0], alpha=0.3)
    ax[2].loglog()
    
    for axis in ax: axis.grid(which='both')
    plt.tight_layout()
    return fig



def datafig(ic, fin, data, bs, dnoise, shotnoise=None):
    nc = ic.shape[-1]
    k, pic = tools.power(ic[0], boxsize=bs)
    k, pf = tools.power(fin[0], boxsize=bs)
    k, pd = tools.power(data[0], boxsize=bs)
    k, pn = tools.power(1+data[0]-fin[0], boxsize=bs)
    if dnoise is not None:
        k, pn2 = tools.power(1+np.random.normal(0, dnoise, nc**3).reshape(fin.shape)[0], boxsize=bs)

    # plt.plot(k, pd/pf)
    # plt.semilogx()
    fig, axar = plt.subplots(2, 2, figsize=(8, 8))

    im = axar[0, 0].imshow(ic[0].sum(axis=0))
    plt.colorbar(im, ax=axar[0, 0])
    axar[0, 0].set_title('IC')
    im = axar[0, 1].imshow(fin[0].sum(axis=0))
    plt.colorbar(im, ax=axar[0, 1])
    axar[0, 1].set_title('Final')
    im = axar[1, 0].imshow(data[0].sum(axis=0))
    plt.colorbar(im, ax=axar[1, 0])
    axar[1, 0].set_title('Data')
    ax = axar[1]
    ax[1].plot(k, pic, label='IC')
    ax[1].plot(k, pf, label='Final')
    ax[1].plot(k, pd, label='Data')
    ax[1].plot(k, pn, label='Noise')
    ax[1].axhline((bs**3/nc**3))
    if shotnoise is not None: ax[1].axhline(shotnoise, color='k', ls="--")

    ax[1].loglog()
    ax[1].grid(which='both')
    ax[1].legend()
    ax[1].set_xlabel('k (h/Mpc)')
    ax[1].set_ylabel('P(k)')
    plt.suptitle('LPT: Boxsize=%d, Nmesh=%d'%(bs, nc))
    return fig

