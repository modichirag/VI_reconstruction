import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

#import flow_utils as futils
#from flow_utils import ResidualBlock, AffineCoupling, trainable_lu_factorization, \
#  SplineParams, TransferSpectra, SimpleRQSpline, ConditionalSpline3D
from flow_utils import *

class MAFFlow(tf.keras.Model):

    def __init__(self, d, nlayers=5, nunits=32, name=None):
        '''
        Parameters:
        ----------
        d : dimension
        nlayers : number of autoregressive layers, default 5
        nunits : number of units in hidden layers, default 32
        '''
        super(MAFFlow, self).__init__(name=name)
        self.d = d
        self.nlayers = nlayers
        self.nunits = nunits
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))      

        
    def build(self, input_shape):       

        bijectors = []
        scale0, scale1 = tf.Variable(tf.ones(self.d), name='scale0'), tf.Variable(tf.ones(self.d), name='scale1')
        shift0, shift1 = tf.Variable(tf.zeros(self.d), name='shift0'), tf.Variable(tf.zeros(self.d), name='shift1')
    
        bijectors.append(tfb.AffineScalar(shift0, scale0))
        for i in range(self.nlayers):
            print(i)
            if i > self.nlayers//2: bijectors.append(tfb.RealNVP(num_masked=self.d//2, bijector_fn=AffineCoupling(shift_only=False), name='nvpaffine%d'%i))
            else: bijectors.append(tfb.RealNVP(num_masked=self.d//2, bijector_fn=AffineCoupling(), name='nvpaffine%d'%i))
            anet = tfb.AutoregressiveNetwork(params=2, hidden_units=[self.nunits, self.nunits], activation='relu')
            abiject = tfb.MaskedAutoregressiveFlow(anet)
            bijectors.append(abiject)
            bijectors.append(tfb.Permute(np.random.permutation(np.arange(self.d).astype(int))))            

        bijectors.append(tfb.AffineScalar(shift1, scale1))    
        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, self.bijector)
        # Hack to get trainable variables
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables
        print('Built')
 
        
    def call(self, x):
        return self.flow.log_prob(x)



          
    
class SplineFlow(tf.keras.Model):

    def __init__(self, d, nsplines=5, nlayers=2, nbins=32, nunits=32, name=None):
        '''
        Parameters:
        ----------
        d : dimension
        nsplines : number of Spline params layers, default 5
        nlayers : number of layers in every Spline kernel, default 1
        nbins : number of bins, default 32
        nunits : number of units in hidden layers, default 32
        '''
        super(SplineFlow, self).__init__(name=name)
        self.d = d
        self.nsplines = nsplines
        self.nlayers = nlayers
        self.nbins = nbins
        self.nunits = nunits
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))


    def build(self, input_shape):       

        bijectors = []
        scale0, scale1 = tf.Variable(tf.ones(self.d), name='scale0'), tf.Variable(tf.ones(self.d), name='scale1')
        shift0, shift1 = tf.Variable(tf.zeros(self.d), name='shift0'), tf.Variable(tf.zeros(self.d), name='shift1')
        lus = [trainable_lu_factorization(self.d) for i in range(self.nsplines)]
        splines = [SplineParams(self.nbins, nlayers=self.nlayers, kreg=1e-2) for i in range(self.nsplines)]
        #shift_and_scales = [tfb.real_nvp_default_template(hidden_layers=[self.nunits, self.nunits]) for i in range(self.nsplines)]

        bijectors.append(tfb.AffineScalar(shift0, scale0))
        for i in range(self.nsplines):
            print("spline : ", i)
            bijectors.append(tfb.RealNVP(num_masked=self.d//2, bijector_fn=AffineCoupling(), name='nvpaffine%d'%i))
            bijectors.append(tfb.RealNVP(num_masked=self.d//2, bijector_fn=splines[i], name='nvpspline%d'%i))
            bijectors.append(tfb.ScaleMatvecLU(*lus[i], validate_args=True, name="perms_train%d"%i))
            if i%2 == 0: bijectors.append(tfb.Permute(np.random.permutation(np.arange(self.d))))

        bijectors.append(tfb.AffineScalar(shift1, scale1))
        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, bijector=self.bijector)
        # Hack to get trainable variables
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables
        print('Built')
 

    def call(self, x):
        return self.flow.log_prob(x)
          

        
class AffineFlow(tf.keras.Model):
    """This is a normalizing flow using the coupling layers defined
    above."""
    def __init__(self, d, name=None):
        super().__init__(name=name)
        self.d=d
        self.noise = tfd.MultivariateNormalDiag(tf.zeros(self.d),tf.ones(self.d))
# 
    def build(self, input_shape):
        self.chain = tfb.Chain([
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff1'), name="nvp1"),
            tfb.Permute(np.arange(self.d)[::-1]),
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff2'), name="nvp2"),
            tfb.Permute(np.arange(self.d)[::-1]),
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff3',
                                                         shift_only=False), name="nvp3"),
            tfb.Permute(np.arange(self.d)[::-1]),
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff4',
                                                         shift_only=False), name="nvp4"),
            tfb.Permute(np.arange(self.d)[::-1]),
        ])       
        
        self.flow = tfd.TransformedDistribution(self.noise, bijector=self.chain)
        # Hack to get trainable variables
        self.flow.sample(1)
        self.bijector = self.chain
        self._variable = self.flow.trainable_variables
        print('Built')
        
    def call(self, x):
        return self.flow.log_prob(x)




class TRENF_classic(tf.Module):
    
    def __init__(self, nc, bs=1., fitmean=True, fitnoise=True, nlayers=1, nknots=100, nbins=32, eps=1e-3, name=None, linknots=False, fitscale=True):
        super(TRENF_classic, self).__init__(name=name)
        self.nc = nc
        self.bs = bs
        self.nlayers = nlayers
        self.nknots = nknots
        self.nbins = nbins
        self.eps = eps
        self.linknots = linknots
        self.fitscale = fitscale
        if fitmean:
          self.loc = tf.Variable(tf.random.normal([nc, nc, nc])*0, name='loc')
        else:
          self.loc = tf.zeros([nc, nc, nc], name='loc')
        if fitnoise:
          self.std = tf.Variable(tf.random.normal([nc, nc, nc])*0+1., name='std')
        else:
          self.std = tf.ones([nc, nc, nc])
        
        self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.std), 2)
        self.is_built=False
        
    def build(self, input_shape):
        print('Build TRENF')
        nc, nlayers, nknots, nbins = self.nc, self.nlayers, self.nknots, self.nbins
        self.tfspecs = [TransferSpectra(nc, nknots, linspace=self.linknots, name='treconv%d/'%i) for i in range(nlayers)]
        self.splines = [SimpleRQSpline(nbins, eps=self.eps, name='spline%d'%i) for i in range(nlayers)]
        
        self.shift = [tf.Variable(0., name='shift%d/'%i) for i in range(nlayers+1)]
        if self.fitscale: self.scale = [tf.Variable(0, name='scale%d/'%i, dtype=tf.float32) for i in range(nlayers+1)]
        else: self.scale = [tf.constant(0.) for i in range(nlayers+1)]

        bijectors = []
        #bijectors.append(tfb.Shift(self.shift[0])(tfb.Scale(log_scale=self.scale[0])))
        for i in range(nlayers):
             bijectors.append(self.tfspecs[i])
             bijectors.append(tfb.Shift(self.shift[i])(tfb.Scale(log_scale=self.scale[i])))
             bijectors.append(self.splines[i])
        bijectors.append(tfb.Shift(self.shift[-1])(tfb.Scale(log_scale=self.scale[-1])))
             
        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, self.bijector)
        #get trainable variables
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables
        

    def __call__(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.log_prob(x)

    def log_prob(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        if not self.is_built:
            if sample_shape is None: 
                print('Give sample shape to build the module')
            else: 
                self.build(sample_shape)
                self.is_built = True
        return self.flow.sample(n)

    def forward(self, z):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.forward(z)

    def inverse(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.inverse(x)






class TRENF_affine(tf.Module):
    
    def __init__(self, nc, bs=1., fitmean=True, fitnoise=False, nlayers=1, nknots=100, eps=1e-3, name=None, meanfield=True, linknots=False, fitscale=True):
        super(TRENF_affine, self).__init__(name=name)
        self.nc = nc
        self.bs = bs
        self.nlayers = nlayers
        self.nknots = nknots
        self.eps = eps
        self.meanfield = meanfield
        self.linknots = linknots
        self.fitscale = fitscale
        if fitmean:
          self.loc = tf.Variable(tf.random.normal([nc, nc, nc])*0, name='loc')
        else:
          self.loc = tf.zeros([nc, nc, nc], name='loc')
        if fitnoise:
          self.std = tf.Variable(tf.random.normal([nc, nc, nc])*0+1., name='std')
        else:
          self.std = tf.ones([nc, nc, nc])
        
        self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.std), 2)
        self.is_built=False
        #else:
        ##    self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=tf.zeros((nc, nc, nc, )), \
        #                                                            scale_diag=tf.ones((nc, nc, nc, ))*nc**1.5 / bs**1.5), 2)
        
    def build(self, input_shape):
        print('Build TRENF')
        nc, nlayers, nknots = self.nc, self.nlayers, self.nknots
        self.tfspecs = [TransferSpectra(nc, nknots, linspace=self.linknots, name='treconv%d/'%i) for i in range(nlayers)]
        if self.meanfield:
          self.shift = [tf.Variable(tf.zeros([1, nc, nc, nc]), name='shift%d/'%i) for i in range(nlayers+1)]
          if self.fitscale : 
            self.scale = [tf.Variable(tf.zeros([1, nc, nc, nc]), name='scale%d/'%i, dtype=tf.float32) for i in range(nlayers+1)]
          else:
            self.scale = [tf.constant(0.) for i in range(nlayers+1)]
        else: 
          self.shift = [tf.Variable(0., name='shift%d/'%i) for i in range(nlayers+1)]
          if self.fitscale:
            self.scale = [tf.Variable(0., name='scale%d/'%i, dtype=tf.float32) for i in range(nlayers+1)]
          else:
            self.scale = [tf.constant(0.) for i in range(nlayers+1)]
          

        bijectors = []
        for i in range(nlayers):
             bijectors.append(tfb.Shift(self.shift[i])(tfb.Scale(log_scale=self.scale[i])))
             bijectors.append(self.tfspecs[i])
        bijectors.append(tfb.Shift(self.shift[-1])(tfb.Scale(log_scale=self.scale[-1])))
             
        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, self.bijector)
        #get trainable variables
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables
        

    def __call__(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.log_prob(x)

    def log_prob(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        if not self.is_built:
            if sample_shape is None: 
                print('Give sample shape to build the module')
            else: 
                self.build(sample_shape)
                self.is_built = True
        return self.flow.sample(n)

    def forward(self, z):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.forward(z)

    def inverse(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.inverse(x)




class TRENF_affine_spline(tf.Module):
    
    def __init__(self, nc, bs=1., fitmean=True, fitnoise=False, nlayers=1, nknots=100, nbins=32, eps=1e-3, name=None, meanfield=True, linknots=False):
        super(TRENF_affine_spline, self).__init__(name=name)
        self.nc = nc
        self.bs = bs
        self.nlayers = nlayers
        self.nknots = nknots
        self.nbins = nbins
        self.eps = eps
        self.meanfield = meanfield
        self.linknots = linknots
        if fitmean:
          self.loc = tf.Variable(tf.random.normal([nc, nc, nc])*0, name='loc')
        else:
          self.loc = tf.zeros([nc, nc, nc], name='loc')
        if fitnoise:
          self.std = tf.Variable(tf.random.normal([nc, nc, nc])*0+1., name='std')
        else:
          self.std = tf.ones([nc, nc, nc])
        
        self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.std), 2)
        self.is_built=False
        #else:
        ##    self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=tf.zeros((nc, nc, nc, )), \
        #                                                            scale_diag=tf.ones((nc, nc, nc, ))*nc**1.5 / bs**1.5), 2)
        
    def build(self, input_shape):
        print('Build TRENF')
        nc, nlayers, nknots, nbins = self.nc, self.nlayers, self.nknots, self.nbins
        self.tfspecs = [TransferSpectra(nc, nknots, linspace=self.linknots, name='treconv%d/'%i) for i in range(nlayers)]
        self.splines = [SimpleRQSpline(nbins, eps=self.eps, name='spline%d'%i) for i in range(nlayers)]
        if self.meanfield:
          self.shift = [tf.Variable(tf.zeros([nc, nc, nc]), name='shift%d/'%i) for i in range(nlayers+1)]
          self.scale = [tf.Variable(tf.zeros([nc, nc, nc]), name='scale%d/'%i, dtype=tf.float32) for i in range(nlayers+1)]
        else: 
          self.shift = [tf.Variable(0., name='shift%d/'%i) for i in range(nlayers+1)]
          self.scale = [tf.Variable(0., name='scale%d/'%i, dtype=tf.float32) for i in range(nlayers+1)]
          

        bijectors = []
        for i in range(nlayers):
             bijectors.append(self.tfspecs[i])
             bijectors.append(tfb.Shift(self.shift[i])(tfb.Scale(log_scale=self.scale[i])))
             bijectors.append(self.splines[i])
        bijectors.append(tfb.Shift(self.shift[-1])(tfb.Scale(log_scale=self.scale[-1])))
             
        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, self.bijector)
        #get trainable variables
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables
        

    def __call__(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.log_prob(x)

    def log_prob(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        if not self.is_built:
            if sample_shape is None: 
                print('Give sample shape to build the module')
            else: 
                self.build(sample_shape)
                self.is_built = True
        return self.flow.sample(n)

    def forward(self, z):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.forward(z)

    def inverse(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.inverse(x)





class Conditional_TRENF_affine(tf.Module):

    def __init__(self, nc, condition, bs=1., fitmean=True, fitnoise=True, nlayers=1, nknots=100, nbins=32, eps=1e-3, name=None):
        super(Conditional_TRENF_affine, self).__init__(name=name)
        self.nc = nc
        self.bs = bs
        self.nlayers = nlayers
        self.nknots = nknots
        self.nbins = nbins
        self.eps = eps
        if fitmean:
            self.loc = tf.Variable(tf.random.normal([nc, nc, nc])*0, name='loc')
        else:
            self.loc = tf.zeros([nc, nc, nc], name='loc')
        if fitnoise:
          self.std = tf.Variable(tf.random.normal([nc, nc, nc])*0+1., name='std')
        else:
          self.std = tf.ones([nc, nc, nc])
        self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.std), 2)
        self.condition = condition
        self.is_built = False
        
        
    def build(self, input_shape):
        print('Build Conditional TRENF')
        print(input_shape)
        nc, nlayers, nknots, nbins = self.nc, self.nlayers, self.nknots, self.nbins
        
        self.tfspecs = [TransferSpectra(nc, nknots, linspace=False, name='treconv%d/'%i) for i in range(nlayers)]

        self.shift = [ConditionalShift(self.condition, nlayers=1, name='cshift%d'%i) for i in range(nlayers+1)]
        self.scale = [ConditionalScale(self.condition, nlayers=1, name='cscale%d'%i) for i in range(nlayers+1)]

        bijectors = []
        for i in range(nlayers):
            bijectors.append(self.scale[i])
            bijectors.append(self.shift[i])
            bijectors.append(self.tfspecs[i])
        bijectors.append(self.scale[-1])
        bijectors.append(self.shift[-1])

        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, self.bijector)
        #get trainable variables
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables

    def __call__(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.log_prob(x)

    def log_prob(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        if not self.is_built:
            if sample_shape is None: 
                print('Give sample shape to build the module')
            else: 
                self.build(sample_shape)
                self.is_built = True
        return self.flow.sample(n)

    def forward(self, z):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.forward(z)

    def inverse(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.inverse(x)




class Conditional_TRENF_Spline(tf.Module):

    def __init__(self, nc, condition, bs=1., fitmean=True, fitnoise=True, nlayers=1, nknots=100, nbins=32, eps=1e-3, name=None):
        super(Conditional_TRENF_Spline, self).__init__(name=name)
        self.nc = nc
        self.bs = bs
        self.nlayers = nlayers
        self.nknots = nknots
        self.nbins = nbins
        self.eps = eps
        if fitmean:
            self.loc = tf.Variable(tf.random.normal([nc, nc, nc])*0, name='loc')
        else:
            self.loc = tf.zeros([nc, nc, nc], name='loc')
        if fitnoise:
          self.std = tf.Variable(tf.random.normal([nc, nc, nc])*0+1., name='std')
        else:
          self.std = tf.ones([nc, nc, nc])
        self.noise = tfd.Independent(tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=self.std), 2)
        self.condition = condition
        self.is_built = False
        
        
    def build(self, input_shape):
        print('Build Conditional TRENF')
        print(input_shape)
        nc, nlayers, nknots, nbins = self.nc, self.nlayers, self.nknots, self.nbins
        
        self.tfspecs = [TransferSpectra(nc, nknots, linspace=False, name='treconv%d/'%i) for i in range(nlayers)]
        #self.splines = [SimpleRQSpline(nbins, eps=self.eps, name='spline%d'%i) for i in range(nlayers)]
        #self.splines = [SplineParams(nbins, name='spline%d'%i) for i in range(nlayers)]
        self.splines = [ConditionalSpline3D(nbins, self.condition, nlayers=1, name='spline%d'%i) for i in range(nlayers)]

        self.shift = [tf.Variable(0., name='shift%d/'%i) for i in range(nlayers+2)]
        self.scale = [tf.Variable(0, name='scale%d/'%i, dtype=tf.float32) for i in range(nlayers+2)]

        bijectors = []
        bijectors.append(tfb.Shift(self.shift[0])(tfb.Scale(log_scale=self.scale[0])))
        for i in range(nlayers):
            bijectors.append(self.tfspecs[i])
            bijectors.append(tfb.Shift(self.shift[i+1])(tfb.Scale(log_scale=self.scale[i+1])))
            bijectors.append(self.splines[i])
        bijectors.append(tfb.Shift(self.shift[-1])(tfb.Scale(log_scale=self.scale[-1])))

        self.bijector = tfb.Chain(bijectors)
        self.flow = tfd.TransformedDistribution(self.noise, self.bijector)
        #get trainable variables                                                                                                                                                                                                                                                                                       
        self.flow.sample(1)
        self._variable = self.flow.trainable_variables

    def __call__(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.log_prob(x)

    def log_prob(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        if not self.is_built:
            if sample_shape is None: 
                print('Give sample shape to build the module')
            else: 
                self.build(sample_shape)
                self.is_built = True
        return self.flow.sample(n)

    def forward(self, z):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.forward(z)

    def inverse(self, x):
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True
        return self.flow.inverse(x)



