import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import flowpm



##############################################
class DM_config():
    

    def __init__(self, evolve, data, dnoise):
        self.evolve = evolve
        self.data = data
        self.dnoise = dnoise

    @tf.function
    def unnormalized_log_prob(self, z, noiselevel=tf.constant(1.)):
        #Chisq
        final_field = self.evolve.pmz(z)
        loglik = tf.reduce_sum(tfd.Normal(final_field, dnoise*noiselevel).log_prob(self.data), axis=(-3, -2, -1))
        #Prior
        logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(z), axis=(-3, -2, -1)) 
        toret = loglik + logprior
        return  toret


    @tf.function
    def grad_log_prob(self, linear, noiselevel=tf.constant(1.)):
        with tf.GradientTape() as tape:
            tape.watch(linear)
            logposterior = tf.reduce_sum(self.unnormalized_log_prob(linear, noiselevel))
        grad = tape.gradient(logposterior, linear)
        return grad



class DM_fourier():

    def __init__(self, evolve, data, dnoise):
        self.evolve = evolve
        self.data = data
        self.dnoise = dnoise

    @tf.function
    def unnormalized_log_prior(self, zk):
        zscale = 1. #(nc**3/2)**0.5
        logprior = tf.reduce_sum(tfd.Normal(0, zscale).log_prob(zk), axis=(-4, -3, -2, -1)) 
        return logprior

    @tf.function
    def unnormalized_log_likelihood(self, lin, noiselevel=tf.constant(1.)):
        #Chisq
        final_field = self.evolve.pm(lin)
        loglik = tf.reduce_sum(tfd.Normal(final_field, self.dnoise*noiselevel).log_prob(self.data), axis=(-3, -2, -1))
        return loglik

    @tf.function
    def unnormalized_log_prob(self, zk, noiselevel=tf.constant(1.)):
        #This function will not have a gradient due to RFFT
        logprior = self.unnormalized_log_prior(zk)
        link = self.evolve.zk_to_link(zk) 
        lin = tf.signal.irfft3d(tf.complex(link[..., 0], link[..., 1]))
        loglik = self.unnormalized_log_likelihood(lin, noiselevel)
        logprob = logprior + loglik
        return logprob

        
    @tf.function
    def grad_log_prob(self, zk, noiselevel):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(zk)
            link = self.evolve.zk_to_link(zk) 
            logprior = self.unnormalized_log_prior(zk)

        lin = tf.signal.irfft3d(tf.complex(link[..., 0], link[..., 1]))
        with tf.GradientTape() as tape2:
            tape2.watch(lin)
            loglik = self.unnormalized_log_likelihood(lin)

        glin = tape2.gradient(loglik, lin)
        glink = tf.signal.rfft3d(glin)
        fac = np.ones(glink.shape)*2
        fac[:, :, :, 0] = 1.
        glink = glink * fac / self.evolve.nc**3
        #glink = 2 * tf.signal.rfft3d(glin) /nc**3  #2 because both + and -k contribute, nc**3 balances nc**1.5 in zk_to_link
        glinkz0, glinkz1 = tf.math.real(glink), tf.math.imag(glink)
        glinkz = tf.stack([glinkz0, glinkz1], -1)    
        gzlik = tape.gradient(link, zk, glinkz)
        gzprior = tape.gradient(logprior, zk)
        grad = gzlik + gzprior
        loss = logprior + loglik

        return grad



class Kwts():

    def __init__(self, evolve, mode, knoise=None):
        self.evolve = evolve
        self.bs, self.nc = evolve.bs, evolve.nc
        self.mode = mode
        if knoise is None: knoise = 1.
        self.knoise = knoise
        self._build()

    def _build(self):
        bs, nc = self.bs, self.nc
        knoise = self.knoise
        if self.mode == 1:
            kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
            kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
            kwts = kmesh**2
            kwts[0, 0, 0] = kwts[0, 0, 1]
            kwts /= kwts[0, 0, 0] 
            self.kwts = np.stack([kwts, kwts], axis=-1)
        if self.mode == 2:
            #ikwts
            kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
            kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
            kwts = kmesh**2
            kwts[0, 0, 0] = kwts[0, 0, 1]
            kwts /= kwts[0, 0, 0] 
            kwts = np.stack([kwts, kwts], axis=-1)
            self.kwts = 1/kwts
        if self.mode == 3:
            kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
            kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
            kwts = kmesh**3
            kwts[0, 0, 0] = kwts[0, 0, 1]
            self.kwts = np.stack([kwts, kwts], axis=-1)
        if self.mode == 4:
            kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
            kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
            kwts = kmesh**3
            kwts[0, 0, 0] = kwts[0, 0, 1]
            mask = kmesh > knoise
            kwts[mask] = knoise**3
            self.kwts = np.stack([kwts, kwts], axis=-1)
            self.kwts /= knoise**3
        if self.mode == 5:
            kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
            kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
            kwts = kmesh**3
            kwts[0, 0, 0] = kwts[0, 0, 1]
            mask = kmesh > knoise
            kwts[mask] = knoise**3
            kwts = np.stack([kwts, kwts], axis=-1)
            kwts /= knoise**3
            self.kwts = 1/kwts
        if self.mode == 6:
            kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=True)
            kmesh = (sum((k*nc/bs)**2 for k in kvec)**0.5)
            kwts = kmesh**3
            kwts[0, 0, 0] = kwts[0, 0, 1]
            mask = kmesh > knoise
            kwts[mask] = knoise**3
            self.kwts = np.stack([kwts, kwts], axis=-1)
        if self.mode == 7:
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
            self.kwts = kwts
        else: 
            self.kwts = None




class Halo_config():
    
    def __init__(self, evolve, data, errormesh, bias):
        self.evolve = evolve
        self.data = data
        self.errormesh = errormesh
        self.bias = bias


    @tf.function
    def unnormalized_log_prob(self, z, Rsm=None):

        bs, nc = self.evolve.bs, self.evolve.nc
        linear = self.evolve.z_to_lin(z)
        bmodel = self.evolve.biasfield(linear, self.bias)
        residual = bmodel - self.data

        if Rsm is not None :
            print("\nAdd annealing section to graph\n")
            Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
            smwts = tf.exp(tf.multiply(-self.evolve.kmesh**2, Rsmsq))
            residualk = flowpm.utils.r2c3d(residual)
            residualk = tf.multiply(residualk, tf.cast(smwts, tf.complex64))
            residual = flowpm.utils.c2r3d(residualk)   

        resk = flowpm.utils.r2c3d(residual, norm=nc**3)
        reskmesh = tf.square(tf.cast(tf.abs(resk), tf.float32))
        chisq = tf.reduce_sum(tf.multiply(reskmesh, 1/self.errormesh), axis=(-3, -2, -1))
        loglik = -chisq* bs**3#/nc**1.5       
        #loglik = tf.reduce_sum(tfd.Normal(bmodel, errormesh).log_prob(self.data), axis=(-3, -2, -1))
        #Prior
        logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(z), axis=(-3, -2, -1)) 
        print("loglik and logp : ", loglik, logprior)
        print(loglik.shape, logprior.shape)
        toret = loglik + logprior
        return  toret


    @tf.function
    def grad_log_prob(self, z, Rsm=None):
        with tf.GradientTape() as tape:
            tape.watch(z)
            logprob = self.unnormalized_log_prob(z, Rsm)
        grad = tape.gradient(logprob, z)
        return grad


    @tf.function
    def val_and_grad_log_prob(self, z, Rsm):
        with tf.GradientTape() as tape:
            tape.watch(z)
            logprob = self.unnormalized_log_prob(z, Rsm)
        grad = tape.gradient(logprob, z)
        return logprob, grad



