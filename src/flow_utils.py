import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.python.keras.layers import ReLU, LeakyReLU
tfd = tfp.distributions
tfb = tfp.bijectors

class ResidualBlock(tf.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(self,
                 nunits=8,
                 activation=tf.nn.relu,
                 dropout_probability=0.,
                 use_batch_norm=False,
                 zero_initialization=True, name=''):
        super().__init__()
        self.activation = activation
        self.linear_layers = [tf.keras.layers.Dense(nunits, name=name + 'resblock_%d'%i) for i in range(2)]
#         self.dropout = nn.Dropout(p=dropout_probability)
#         if zero_initialization:
#             init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
#             init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def __call__(self, inputs, context=None):
        temps = inputs       
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        temps = self.activation(temps)
#         temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        return inputs + temps



def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
    with tf.name_scope(name or 'trainable_lu_factorization'):
        event_size = tf.convert_to_tensor(
            event_size, dtype_hint=tf.int32, name='event_size')
        batch_shape = tf.convert_to_tensor(
            batch_shape, dtype_hint=event_size.dtype, name='batch_shape')
        random_matrix = tf.random.uniform(
            shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
            dtype=dtype,
            seed=seed)
        random_orthonormal = tf.linalg.qr(random_matrix)[0]
        lower_upper, permutation = tf.linalg.lu(random_orthonormal)
        lower_upper = tf.Variable(
            initial_value=lower_upper,
            trainable=True,
            name='lower_upper')
        permutation = tf.Variable(
            initial_value=permutation,
            trainable=False,
            name='permutation')
        return lower_upper, permutation



    
class AffineCoupling(tf.Module):
    """This is the coupling layer used in the Flow."""
  
    def __init__(self, nunits=128, nlayers=2, activation=tf.nn.leaky_relu, shift_only=True, **kwargs):
        super().__init__(**kwargs)
        self.shift_only = shift_only
        self.nunits = nunits
        self.nlayers = nlayers
        self.activation = activation
        self.build = True
        
    def __call__(self, x, output_units, **condition_kwargs):
        if self.build:
            self.linear_layers = [tf.keras.layers.Dense(self.nunits, name='affdense%d'%i) for i in range(self.nlayers)]
            self.shiftlayer = tf.keras.layers.Dense(output_units, name='shiftlayer')
            if not self.shift_only: self.scalelayer = tf.keras.layers.Dense(output_units, name='scalelayer')
            self.build = False

        net = x
        for i in range(self.nlayers):
            net = self.linear_layers[i](net)
            net = self.activation(net)
        shifter = tfb.Shift(self.shiftlayer(net))
        if self.shift_only:
          return shifter
        else:
          scaler = tfb.Scale(tf.clip_by_value(tf.nn.softplus(self.scalelayer(net)), 1e-2, 1e1))
          return tfb.Chain([shifter, scaler])
    

class AffineCoupling_residual(tf.Module):
    """This is the coupling layer used in the Flow."""

    def __init__(self, nunits=32, nlayers=2, activation=tf.nn.leaky_relu, shift_only=True, **kwargs):
        super().__init__(**kwargs)
        self.shift_only = shift_only
        self.nunits = nunits
        self.nlayers = nlayers
        self.activation = activation
        self.build = True

    def __call__(self, x, output_units, **condition_kwargs):
        if self.build:
            self.linear_layers = [tf.keras.layers.Dense(self.nunits, name='affdense')] + \
                                [ResidualBlock(self.nunits, name='affres%d'%i) for i in range(self.nlayers)]
            self.shiftlayer = tf.keras.layers.Dense(output_units, name='shiftlayer')
            if not self.shift_only: self.scalelayer = tf.keras.layers.Dense(output_units, name='scalelayer')
            self.build = False

        net = x
        for i in range(self.nlayers):
            print(net.shape)
            net = self.linear_layers[i](net)
            print(net.shape)
            net = self.activation(net)
        shifter = tfb.Shift(self.shiftlayer(net))
        if self.shift_only:
          return shifter
        else:
          scaler = tfb.Scale(tf.clip_by_value(tf.nn.softplus(self.scalelayer(net)), 1e-2, 1e1))
          return tfb.Chain([shifter, scaler])
    
    
class SplineParams(tf.Module):

    def __init__(self, nbins=32, nlayers=2, kreg=1e-3, name='spline'):
      
        super().__init__(name=name)
        self._nbins = nbins
        self.is_built = False
        self._bin_widths = None
        self._bin_heights = None
        self._knot_slopes = None
        self.kreg = kreg
        self.nlayers = nlayers
        
    def __call__(self, x, nunits=None):
        if not self.is_built:
            print('Build Spline')

            if nunits is None:
                self.w = [tf.keras.layers.Dense(self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-w'%i) for i in range(self.nlayers)]
                self.h = [tf.keras.layers.Dense(self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-h'%i) for i in range(self.nlayers)]
                self.s = [tf.keras.layers.Dense((self._nbins-1), kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-s'%i) for i in range(self.nlayers)]
            else:
                self.w = [tf.keras.layers.Dense(nunits * self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-w'%i) for i in range(self.nlayers)]
                self.h = [tf.keras.layers.Dense(nunits * self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-h'%i) for i in range(self.nlayers)]
                self.s = [tf.keras.layers.Dense(nunits * (self._nbins-1), kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-s'%i) for i in range(self.nlayers)]
                
            def _widths(x):
                net = tf.identity(x)
                for i in range(self.nlayers):
                    net = self.w[i](net)
                    net = tf.nn.leaky_relu(net)
                if nunits is not None:
                    out_shape = tf.concat((tf.shape(net)[:-1], (nunits, self._nbins)), 0)
                    net = tf.reshape(net, out_shape)
                return tf.math.softmax(net, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

            def _heights(x):
                net = tf.identity(x)
                for i in range(self.nlayers):
                    net = self.h[i](net)
                    net = tf.nn.leaky_relu(net)
                if nunits is not None:
                    out_shape = tf.concat((tf.shape(net)[:-1], (nunits, self._nbins)), 0)
                    net = tf.reshape(net, out_shape)
                return tf.math.softmax(net, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

            def _slopes(x):
                net = tf.identity(x)
                for i in range(self.nlayers):
                    net = self.s[i](net)
                    net = tf.nn.leaky_relu(net)
                if nunits is not None:
                    out_shape = tf.concat((tf.shape(net)[:-1], (nunits, self._nbins - 1)), 0)
                    net = tf.reshape(net, out_shape)
                return tf.math.softplus(net) + 1e-2

            self._bin_widths = _widths
            self._bin_heights = _heights
            self._knot_slopes = _slopes
#             self.B = _B                                                                                                                                                                                                                                                                                                     
            self.is_built = True

        return tfb.RationalQuadraticSpline(
            bin_widths=self._bin_widths(x),
            bin_heights=self._bin_heights(x),
            knot_slopes=self._knot_slopes(x), range_min=-1)


##class SplineParams(tf.Module):
##
##    def __init__(self, nbins=32, nlayers=2, kreg=1e-3):
##        self._nbins = nbins
##        self._built = False
##        self._bin_widths = None
##        self._bin_heights = None
##        self._knot_slopes = None   
##        self.kreg = kreg
##        self.nlayers = nlayers
##    
##    
##    def __call__(self, x, nunits):
##        if not self._built:
##            print('Build')
##
##            self.w = [tf.keras.layers.Dense(nunits * self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg)) for _ in range(self.nlayers)]
##            self.h = [tf.keras.layers.Dense(nunits * self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg)) for _ in range(self.nlayers)]
##            self.s = [tf.keras.layers.Dense(nunits * (self._nbins-1), kernel_regularizer=tf.keras.regularizers.L2(self.kreg)) for _ in range(self.nlayers)]
##            
##            
##            def _widths(x):
##                net = tf.identity(x)
##                for i in range(self.nlayers):
##                    net = self.w[i](net)
##                    net = tf.nn.leaky_relu(net)
##                out_shape = tf.concat((tf.shape(net)[:-1], (nunits, self._nbins)), 0)
##                net = tf.reshape(net, out_shape)
##                return tf.math.softmax(net, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2
##
##            def _heights(x):
##                net = tf.identity(x)
##                for i in range(self.nlayers):
##                    net = self.h[i](net)
##                    net = tf.nn.leaky_relu(net)
##                out_shape = tf.concat((tf.shape(net)[:-1], (nunits, self._nbins)), 0)
##                net = tf.reshape(net, out_shape)
##                return tf.math.softmax(net, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2
##
##            def _slopes(x):
##                net = tf.identity(x)
##                for i in range(self.nlayers):
##                    net = self.s[i](net)
##                    net = tf.nn.leaky_relu(net)
##                out_shape = tf.concat((tf.shape(net)[:-1], (nunits, self._nbins - 1)), 0)
##                net = tf.reshape(net, out_shape)
##                return tf.math.softplus(net) + 1e-2
##
##            self._bin_widths = _widths
##            self._bin_heights = _heights
##            self._knot_slopes = _slopes
###             self.B = _B
##            self._built = True
##
##        return tfb.RationalQuadraticSpline(
##            bin_widths=self._bin_widths(x),
##            bin_heights=self._bin_heights(x),
##            knot_slopes=self._knot_slopes(x), range_min=-1)
##



class CubicHermite(tf.Module):
    """
    """

    def __init__(self, x, y, dydx, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.dydx = dydx

    def __call__(self, x, index=None):
        if index is None: 
            if len(x.shape) != len(self.x.shape):
                newshape = x.shape[:-1] + self.x.shape
                xindex = tf.broadcast_to(self.x, newshape)
            else: xindex = self.x
            index = tf.searchsorted(xindex, x) - 1
            
        t = x - tf.gather(self.x, index)
        yi = tf.gather(self.y, index)
        yi1 = tf.gather(self.y, index+1)
        dyi = tf.gather(self.dydx, index)
        dyi1 = tf.gather(self.dydx, index+1)
        a = yi
        b = dyi
        c = 3*(yi1 - yi) - 2*dyi - dyi1
        d = 2*(yi - yi1) + dyi + dyi1
        
        y = c + t*d
        y = b + t*y
        y = a + t*y
        
        return y
        
        


def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return kvector given a shape (nc, nc, nc) and boxsize 
    """
    k = []
    for d in range(len(shape)):
        kd = np.fft.fftfreq(shape[d])
        kd *= 2 * np.pi / boxsize * shape[d]
        kdshape = np.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k


class TransferSpectra(tfb.Bijector):

    def __init__(self, nc, nknot, x=None, y=None, dy=None, y0=None, k=None, normalize=True, 
                 positivetf=True, linspace=True, zeromode=True, normed=False, 
                 forward_min_event_ndims=3, validate_args=False, name='treconv/'):
        super().__init__(
          validate_args=validate_args,
          forward_min_event_ndims=forward_min_event_ndims, name=name) #Forward_min_events is set to default at 3D
        
        #setup k field
        self.nc = nc
        self.normed = normed
        if k is None:
            kvec = fftk((nc, nc, nc),  boxsize=1, symmetric=False)
            delkv = kvec[0].flatten()[1] - kvec[0].flatten()[0]
            self.k = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
            self.k /= self.k.max()
        else: 
            self.k = k
        if x is None:
            if linspace: self.x = np.linspace(1/nc, 1., nknot, endpoint=True, dtype=np.float32)
            else: self.x = np.logspace(np.log10(1/nc), 0., nknot, endpoint=True, dtype=np.float32)
        else: 
            self.x = x 
        
        self._find_kindex()            
        
        #setup interpolation
        if y is None: self.y = tf.Variable(tf.ones(self.x.shape)* 0. + np.log(np.exp(1.)-1.), name=name+'y')
        else: self.y = y
        if dy is None: self.dy = tf.Variable(tf.zeros(self.x.shape), name=name+'dy')
        else: self.dy = dy
        #if y0 is None: self.y0 = tf.Variable(1., dtype=tf.float32)
        if y0 is None: self.y0 = tf.Variable(np.log(np.exp(1.)-1.), dtype=tf.float32, name=name+'y0')
        else: self.y0 = y0 
        self.positivetf = positivetf
        self.tfk = CubicHermite(self.x, self.y, self.dy)
        
    def _normalize_x(self):
        delx = self.x[-1] - self.x[0] 
        self.x = self.x/delx
        self.k = self.k/delx
        
        
    def _find_kindex(self):
        self.kindex = np.searchsorted(self.x, self.k)- 1
        

        
    def get_tfk(self):
        tfk = self.tfk(self.k, self.kindex)
        tfk = tfk + tf.scatter_nd([[0, 0, 0]], [self.y0], tfk.shape)
        if self.positivetf: tfk = tf.nn.softplus(tfk)
        #if self.positivetf: tfk = tf.nn.leaky_relu(tfk)
        return tfk
        
        
    def _forward(self, x, dtype=tf.float32, cdtype=tf.complex64):
        xk = tf.signal.fft3d(tf.cast(x, cdtype))
        if self.normed: xk = xk/self.nc**1.5
        tfk = self.get_tfk()
        xk = xk * tf.cast(tfk, cdtype)
        if self.normed: xk = xk*self.nc**1.5
        x = tf.cast(tf.signal.ifft3d(xk), dtype)
        return x 

    def _inverse(self, y, dtype=tf.float32, cdtype=tf.complex64):
        yk = tf.signal.fft3d(tf.cast(y, cdtype))
        if self.normed: yk = yk/self.nc**1.5
        tfk = self.get_tfk()
        yk = yk / tf.cast(tfk, cdtype)
        if self.normed: yk = yk/self.nc**1.5
        y = tf.cast(tf.signal.ifft3d(yk), dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        tfk = self.get_tfk()
#         return tf.math.log(tf.abs(tfk))
        return tf.reduce_sum(tf.math.log(tf.abs(tfk)))




class ComplexScale(tfb.Bijector):

    def __init__(self, shape, dtype=tf.complex64,  validate_args=False, forward_min_event_ndims=3, name='complexscale/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name) #Forward_min_events is set to default at 3D
        
        self.shape = shape
        self.nc = np.prod(shape)**(1/3.)
        self.rsigma = tf.Variable(tf.ones(shape), name=name+"r")
        self.isigma = tf.Variable(tf.zeros(shape), name=name+"i")
        #self.sigma = tf.Variable(tf.cast(tf.ones(shape), dtype), name=name)
       
    def _forward(self, x, dtype=tf.float32, cdtype=tf.complex64):
        xk = tf.signal.fft3d(tf.cast(x, cdtype))
        xk = xk/self.nc**1.5
        sigma = tf.complex(self.rsigma, self.isigma)
        xk = xk * sigma
        xk = xk*self.nc**1.5
        x = tf.cast(tf.signal.ifft3d(xk), dtype)
        return x 

    def _inverse(self, y, dtype=tf.float32, cdtype=tf.complex64):
        yk = tf.signal.fft3d(tf.cast(y, cdtype))
        yk = yk/self.nc**1.5
        sigma = tf.complex(self.rsigma, self.isigma)
        yk = yk / sigma
        yk = yk*self.nc**1.5
        y = tf.cast(tf.signal.ifft3d(yk), dtype)
        return y

    def _forward_log_det_jacobian(self, x):
#         return tf.math.log(tf.abs(tfk))
        sigma = tf.complex(self.rsigma, self.isigma)
        return tf.reduce_sum(tf.math.log(tf.abs(sigma)))


class ComplexShift(tfb.Bijector):

    def __init__(self, shape, dtype=tf.complex64,  validate_args=False, forward_min_event_ndims=3, name='complexshift/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name) #Forward_min_events is set to default at 3D
        self.shape = shape
        self.nc = np.prod(shape)**(1/3.)
        self.rloc = tf.Variable(tf.zeros(shape), name=name+"r")
        self.iloc = tf.Variable(tf.zeros(shape), name=name+"i")
        #self.loc = tf.Variable(tf.cast(tf.ones(shape), dtype), name=name)
       
    def _forward(self, x, dtype=tf.float32, cdtype=tf.complex64):      
        xk = tf.signal.fft3d(tf.cast(x, cdtype))
        xk = xk/self.nc**1.5
        loc = tf.complex(self.rloc, self.iloc)
        xk = xk + loc
        xk = xk*self.nc**1.5
        y = tf.cast(tf.signal.ifft3d(xk), dtype)
        return y

    def _inverse(self, y, dtype=tf.float32, cdtype=tf.complex64):
        yk = tf.signal.fft3d(tf.cast(y, cdtype))
        yk = yk/self.nc**1.5
        loc = tf.complex(self.rloc, self.iloc)
        yk = yk - loc
        yk = yk*self.nc**1.5
        x = tf.cast(tf.signal.ifft3d(yk), dtype)
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(1.)


class RealShift(tfb.Bijector):

    def __init__(self, shape, dtype=tf.complex64,  validate_args=False, forward_min_event_ndims=3, name='realshift/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name) #Forward_min_events is set to default at 3D        
        self.loc = tf.Variable(tf.zeros(shape), name=name+"loc")
       
    def _forward(self, x, dtype=tf.float32, cdtype=tf.complex64):
        y = x + self.loc
        return y

    def _inverse(self, y, dtype=tf.float32, cdtype=tf.complex64):
        x = y - self.loc
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(1.)


class ComplexScaleandShift(tfb.Bijector):

    def __init__(self, shape, dtype=tf.complex64,  validate_args=False, forward_min_event_ndims=3, name='complexscaleandshift/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name) #Forward_min_events is set to default at 3D
        
        self.shape = shape
        self.nc = np.prod(shape)**(1/3.)
        self.rscale = tf.Variable(tf.ones(shape), name=name+"rscale")
        self.iscale = tf.Variable(tf.zeros(shape), name=name+"iscale")
        self.rloc = tf.Variable(tf.zeros(shape), name=name+"rloc")
        self.iloc = tf.Variable(tf.zeros(shape), name=name+"iloc")
        #self.scale = tf.Variable(tf.cast(tf.ones(shape), dtype), name=name)
       
    def _forward(self, x, dtype=tf.float32, cdtype=tf.complex64):
        xk = tf.signal.fft3d(tf.cast(x, cdtype))
        xk = xk/self.nc**1.5
        scale = tf.complex(self.rscale, self.iscale)
        loc = tf.complex(self.rloc, self.iloc)
        xk = xk*scale + loc
        xk = xk*self.nc**1.5
        x = tf.cast(tf.signal.ifft3d(xk), dtype)
        return x 

    def _inverse(self, y, dtype=tf.float32, cdtype=tf.complex64):
        yk = tf.signal.fft3d(tf.cast(y, cdtype))
        yk = yk/self.nc**1.5
        scale = tf.complex(self.rscale, self.iscale)
        loc = tf.complex(self.rloc, self.iloc)
        yk = (yk-loc)/scale
        yk = yk*self.nc**1.5
        y = tf.cast(tf.signal.ifft3d(yk), dtype)
        return y

    def _forward_log_det_jacobian(self, x):
        sigma = tf.complex(self.rscale, self.iscale)
        return tf.reduce_sum(tf.math.log(tf.abs(sigma)))



class SimpleRQSpline(tfb.Bijector):

    def __init__(self, nbins, validate_args=False, forward_min_event_ndims=0, eps=1e-3, name='rqspline/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name)    

        self._nbins = nbins
        self.eps = eps
        self.w = tf.Variable(tf.ones([nbins])/nbins, name='w-%s/'%self._name)
        self.h = tf.Variable(tf.ones([nbins])/nbins, name='h-%s/'%self._name)
        self.s = tf.Variable(tf.zeros([nbins-1]) + np.log(np.exp(1.) - 1.), name='s-%s/'%self._name)
            

    def _widths(self):
        return tf.math.softmax(self.w) * (2 - self._nbins * self.eps) + self.eps

    def _heights(self):
        return tf.math.softmax(self.h, axis=-1) * (2 - self._nbins * self.eps) + self.eps

    def _slopes(self):
        return tf.math.softplus(self.s) + self.eps


    def _forward(self, x):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._widths(),
            bin_heights=self._heights(),
            knot_slopes=self._slopes(), range_min=-1).forward(x)

    def _inverse(self, x):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._widths(),
            bin_heights=self._heights(),
            knot_slopes=self._slopes(), range_min=-1).inverse(x)
    
    def _forward_log_det_jacobian(self, x):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._widths(),
            bin_heights=self._heights(),
            knot_slopes=self._slopes(), range_min=-1).forward_log_det_jacobian(x, self.forward_min_event_ndims)





class ConditionalSpline3D(tfb.Bijector):

    def __init__(self, nbins, condition, nunits=None, nlayers=2, kreg=0., validate_args=False, forward_min_event_ndims=0, \
                 kernel_size=3, eps=1e-3, name='rqspline/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name)

        self._nbins = nbins
        self.eps = eps
        self.condition = condition
        self.nlayers = nlayers
        self.kreg = kreg
        self.nunits = nunits
        self.kernel_size = kernel_size
        if nunits is None:
            self.w = [Conv3D(self._nbins, kernel_size=self.kernel_size,  kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/%d-w'%i) for i in range(self.nlayers)]
            self.h = [Conv3D(self._nbins, kernel_size=self.kernel_size, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/%d-h'%i) for i in range(self.nlayers)]
            self.s = [Conv3D((self._nbins-1), kernel_size=self.kernel_size, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/%d-s'%i) for i in range(self.nlayers)]
#             self.w = [tf.keras.layers.Dense(self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-w'%i) for i in range(self.nlayers)]
#             self.h = [tf.keras.layers.Dense(self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-h'%i) for i in range(self.nlayers)]
#             self.s = [tf.keras.layers.Dense((self._nbins-1), kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-s'%i) for i in range(self.nlayers)]
        else:
            self.w = [tf.keras.layers.Dense(nunits * self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-w'%i) for i in range(self.nlayers)]
            self.h = [tf.keras.layers.Dense(nunits * self._nbins, kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-h'%i) for i in range(self.nlayers)]
            self.s = [tf.keras.layers.Dense(nunits * (self._nbins-1), kernel_regularizer=tf.keras.regularizers.L2(self.kreg), name=self._name+'/%d-s'%i) for i in range(self.nlayers)]


    def _widths(self):
        net = tf.identity(self.condition)
        for i in range(self.nlayers):
            net = self.w[i](net)
            if i != self.nlayers-1: net = tf.nn.leaky_relu(net)
        if self.nunits is not None:
            out_shape = tf.concat((tf.shape(net)[:-1], (self.nunits, self._nbins)), 0)
            net = tf.reshape(net, out_shape)
        return tf.math.softmax(net, axis=-1) * (2 - self._nbins * self.eps) + self.eps

    def _heights(self):
        net = tf.identity(self.condition)
        for i in range(self.nlayers):
            net = self.h[i](net)
            if i != self.nlayers-1: net = tf.nn.leaky_relu(net)
        if self.nunits is not None:
            out_shape = tf.concat((tf.shape(net)[:-1], (self.nunits, self._nbins)), 0)
            net = tf.reshape(net, out_shape)
        return tf.math.softmax(net, axis=-1) * (2 - self._nbins * self.eps) + self.eps

    def _slopes(self):
        net = tf.identity(self.condition)
        for i in range(self.nlayers):
            net = self.s[i](net)
            if i != self.nlayers-1: net = tf.nn.leaky_relu(net)
        if self.nunits is not None:
            out_shape = tf.concat((tf.shape(net)[:-1], (self.nunits, self._nbins - 1)), 0)
            net = tf.reshape(net, out_shape)
        return tf.math.softplus(net) + self.eps


    def _forward(self, x):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._widths(),
            bin_heights=self._heights(),
            knot_slopes=self._slopes(), range_min=-1).forward(x)


    def _inverse(self, x):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._widths(),
            bin_heights=self._heights(),
            knot_slopes=self._slopes(), range_min=-1).inverse(x)

    def _forward_log_det_jacobian(self, x):
        return tfb.RationalQuadraticSpline(
            bin_widths=self._widths(),
            bin_heights=self._heights(),
            knot_slopes=self._slopes(), range_min=-1).forward_log_det_jacobian(x, self.forward_min_event_ndims)






class ConditionalShift(tfb.Bijector):

    def __init__(self, condition, nchannels=8, nlayers=2, kreg=0., validate_args=False, forward_min_event_ndims=0, \
                 kernel_size=3, eps=1e-3, name='cshift/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name)

        self.condition = condition
        self.nlayers = nlayers
        self.kreg = kreg
        self.kernel_size = kernel_size
        self.transform = [Conv3D(nchannels, kernel_size=self.kernel_size,  kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/cnn%d'%i) for i in range(self.nlayers)]
        self.transform = self.transform + [Conv3D(1, kernel_size=self.kernel_size,  kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/cnnlast')]

    def _shift(self):
        net = tf.identity(self.condition)
        for i in range(self.nlayers):
            net = self.transform[i](net)
            net = tf.nn.leaky_relu(net)
        net = self.transform[-1](net)
        return net[..., 0]



    def _forward(self, x):
        return tfb.Shift(self._shift()).forward(x)

    def _inverse(self, x):
        return tfb.Shift(self._shift()).inverse(x)

    def _forward_log_det_jacobian(self, x):
        return tfb.Shift(self._shift()).forward_log_det_jacobian(x, self.forward_min_event_ndims)




class ConditionalScale(tfb.Bijector):

    def __init__(self, condition, nchannels=8, nlayers=2, kreg=0., validate_args=False, forward_min_event_ndims=0, \
                 kernel_size=3, eps=1e-3, name='cscale/'):
        super().__init__(
          validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name)

        self.condition = condition
        self.nlayers = nlayers
        self.kreg = kreg
        self.kernel_size = kernel_size
        self.transform = [Conv3D(nchannels, kernel_size=self.kernel_size,  kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/cnn%d'%i) for i in range(self.nlayers)]
        self.transform = self.transform + [Conv3D(1, kernel_size=self.kernel_size,  kernel_regularizer=tf.keras.regularizers.L2(self.kreg), padding='SAME', name=self._name+'/cnnlast')]

    def _scale(self):
        net = tf.identity(self.condition)
        for i in range(self.nlayers):
            net = self.transform[i](net)
            net = tf.nn.leaky_relu(net)
        net = self.transform[-1](net)
        return tf.nn.softplus(net[..., 0])



    def _forward(self, x):
        return tfb.Scale(self._scale()).forward(x)

    def _inverse(self, x):
        return tfb.Scale(self._scale()).inverse(x)

    def _forward_log_det_jacobian(self, x):
        return tfb.Scale(self._scale()).forward_log_det_jacobian(x, self.forward_min_event_ndims)

