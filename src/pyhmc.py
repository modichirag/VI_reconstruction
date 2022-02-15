import numpy as np

class DualAveragingStepSize():
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75):
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept

        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
 
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa

        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step

        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)


#$#
#$#
#$#
#$#
#$#class PyHMC():
#$#    
#$#    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None, invmetric=None):
#$#
#$#        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
#$#        self.V = lambda x, y : self.log_prob(x, *y)*-1.
#$#        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
#$#        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
#$#
#$#        digaonal = True
#$#        if invemetric is  None: self.invmetric = 1. 
#$#        else:
#$#            diagonal = False
#$#            self.invmetric = invmetric
#$#        if invmetric_diag is None: self.invmetric_diag = 1. 
#$#        else: self.invmetric_diag = invmetric_diag
#$#
#$#        self.metricstd = self.invmetric_diag**-0.5
#$#
#$#        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
#$#        self.KE_g = lambda p: p * self.invmetric_diag
#$#        self.returnV = returnV
#$#    
#$#    def V_g(self, x, y):
#$#        self.Vgcount += 1
#$#        return self.grad_log_prob(x, *y)*-1.
#$#        
#$#    def unit_norm_KE(self, p):
#$#        return 0.5 * (p**2).sum()
#$#
#$#    def unit_norm_KE_g(self, p):
#$#        return p
#$#
#$#    def H(self, q, p, args):
#$#        self.Hcount += 1
#$#        if self.returnV:
#$#            V = self.V(q, args)
#$#            return V, V + self.KE(p)
#$#        else: return self.V(q, args) + self.KE(p)
#$#
#$#    def leapfrog(self, q, p, N, step_size, args):
#$#        self.leapcount += 1 
#$#        q0, p0 = q, p
#$#        try:
#$#            p = p - 0.5*step_size * self.V_g(q, args) 
#$#            for i in range(N-1):
#$#                q = q + step_size * self.KE_g(p)
#$#                p = p - step_size * self.V_g(q, args) 
#$#            q = q + step_size * self.KE_g(p)
#$#            p = p - 0.5*step_size * self.V_g(q, args) 
#$#            return q, p
#$#        except Exception as e:
#$#            print(e)
#$#            return q0, p0
#$#
#$#    def metropolis(self, qp0, qp1, args):
#$#        q0, p0 = qp0
#$#        q1, p1 = qp1
#$#        if self.returnV:
#$#            V0, H0 = self.H(q0, p0, args)
#$#            V1, H1 = self.H(q1, p1, args)
#$#        else:
#$#            H0 = self.H(q0, p0, args)
#$#            H1 = self.H(q1, p1, args)
#$#            V0, V1 = None, None
#$#        prob = np.exp(H0 - H1)
#$#        #prob = min(1., np.exp(H0 - H1))
#$#        #if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
#$#        if np.isnan(prob) or (q0-q1).sum()==0: 
#$#            return q0, p0, 2., [H0, H1, V0]
#$#        elif np.random.uniform(0., 1., size=1) > min(1., prob):
#$#            return q0, p0, 0., [H0, H1, V0]
#$#        else: return q1, p1, 1., [H0, H1, V1]
#$#
#$#
#$#    def hmc_step(self, q, N, step_size, args=[]):
#$#        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
#$#        p = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
#$#        q1, p1 = self.leapfrog(q, p, N, step_size, args)
#$#        q, p, accepted, energy = self.metropolis([q, p], [q1, p1], args)
#$#        return q, p, accepted, energy, [self.Hcount, self.Vgcount, self.leapcount]
#$#


class PyHMC():
    
    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x, y : self.log_prob(x, *y)*-1.
        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
        self.KE_g = lambda p: p * self.invmetric_diag
        self.returnV = returnV
    
    def V_g(self, x, y):
        self.Vgcount += 1
        return self.grad_log_prob(x, *y)*-1.
        
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q, p, args):
        self.Hcount += 1
        if self.returnV:
            V = self.V(q, args)
            return V, V + self.KE(p)
        else: return self.V(q, args) + self.KE(p)

    def leapfrog(self, q, p, N, step_size, args):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q, args) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q, args) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q, args) 
            return q, p
        except Exception as e:
            print(e)
            return q0, p0

    def metropolis(self, qp0, qp1, args):
        q0, p0 = qp0
        q1, p1 = qp1
        if self.returnV:
            V0, H0 = self.H(q0, p0, args)
            V1, H1 = self.H(q1, p1, args)
        else:
            H0 = self.H(q0, p0, args)
            H1 = self.H(q1, p1, args)
            V0, V1 = None, None
        prob = np.exp(H0 - H1)
        #if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
        if np.isnan(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., [H0, H1, V0]
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., [H0, H1, V0]
        else: return q1, p1, 1., [H0, H1, V1]


    #######CHANGE args[] to NONE########
    def hmc_step(self, q, N, step_size, args=[]):
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        q1, p1 = self.leapfrog(q, p, N, step_size, args)
        q, p, accepted, energy = self.metropolis([q, p], [q1, p1], args)
        return q, p, accepted, energy, [self.Hcount, self.Vgcount, self.leapcount]



class PyHMC_fourier():
    
    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x, y : self.log_prob(x, *y)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.returnV = returnV
    

    def p_init(self, p):
        n = p.shape[-1]
        pc = np.fft.fftn(p/n**1.5)
        pc = pc*self.metricstd
        p = np.fft.ifftn(pc*n**1.5).real
        return p
    
        
    def KE(self, p):
        n = p.shape[-1]
        pc = np.fft.fftn(p/n**1.5)
        ke = (np.abs(pc)**2 * self.invmetric_diag).sum()
        return ke

    def KE_g(self, p):
        n = p.shape[-1]
        pc = np.fft.fftn(p/n**1.5)
        pc_g = pc*self.invmetric_diag
        p_g = 2* np.fft.ifftn(pc_g*n**1.5).real
        return p_g
    

    def V_g(self, x, y):
        self.Vgcount += 1
        return self.grad_log_prob(x, *y)*-1.
        
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q, p, args):
        self.Hcount += 1
        if self.returnV:
            V = self.V(q, args)
            return V, V + self.KE(p)
        else: return self.V(q, args) + self.KE(p)

    def leapfrog(self, q, p, N, step_size, args):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q, args) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q, args) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q, args) 
            return q, p
        except Exception as e:
            print(e)
            return q0, p0

    def metropolis(self, qp0, qp1, args):
        q0, p0 = qp0
        q1, p1 = qp1
        if self.returnV:
            V0, H0 = self.H(q0, p0, args)
            V1, H1 = self.H(q1, p1, args)
        else:
            H0 = self.H(q0, p0, args)
            H1 = self.H(q1, p1, args)
            V0, V1 = None, None
        prob = np.exp(H0 - H1)
        #if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
        if np.isnan(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., [H0, H1, V0]
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., [H0, H1, V0]
        else: return q1, p1, 1., [H0, H1, V1]


    def hmc_step(self, q, N, step_size, args=[]):
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape)
        p = self.p_init(p)
        q1, p1 = self.leapfrog(q, p, N, step_size, args)
        q, p, accepted, energy = self.metropolis([q, p], [q1, p1], args)
        return q, p, accepted, energy, [self.Hcount, self.Vgcount, self.leapcount]


class PyHMC_diff():
    
    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag)
        self.KE_g = lambda p: p * self.invmetric_diag
        self.returnV = returnV
    
    def V_g(self, x):
        self.Vgcount += 1
        return self.grad_log_prob(x)*-1.
        

    def H(self, q,p):
        self.Hcount += 1
        if self.returnV:
            V = self.V(q)
            return V, V + self.KE(p)
        else: return self.V(q) + self.KE(p)

    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            return q, p
        except Exception as e:
            print(e)
            return q0, p0

    def metropolis(self, qp0, qp1):
        q0, p0 = qp0
        q1, p1 = qp1
        if self.returnV:
            V0, H0 = self.H(q0, p0)
            V1, H1 = self.H(q1, p1)
        else:
            H0 = self.H(q0, p0)
            H1 = self.H(q1, p1)
            V0, V1 = None, None
        delH = np.sum(H0-H1, axis=(-3, -2, -1))
        H0, H1 = np.sum(H0, axis=(-3, -2, -1)) ,  np.sum(H1, axis=(-3, -2, -1))   
        delH2 = H0- H1
        prob = np.exp(delH)
        print("floating point : ", delH, delH2, np.exp(delH),  np.exp(delH2))
        #prob = min(1., np.exp(H0 - H1))
        #if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
        if np.isnan(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., [H0, H1, V0]
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., [H0, H1, V0]
        else: return q1, p1, 1., [H0, H1, V1]


    def hmc_step(self, q, N, step_size):
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        q1, p1 = self.leapfrog(q, p, N, step_size)
        q, p, accepted, energy = self.metropolis([q, p], [q1, p1])
        return q, p, accepted, energy, [self.Hcount, self.Vgcount, self.leapcount]

##
class PyHMC_batch():
    
    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.returnV = returnV
    
    def KE(self, p):
        #ke = np.sum(0.5*(p**2), axis=(-3, -2, -1))
        ke = np.sum(0.5*(p**2), axis=tuple(range(1, len(p.shape))))
        #ke = np.sum(0.5*(p**2 * self.invmetric_diag), axis=(-3, -2, -1))
        return ke

    def KE_g(self, p):
        return p 

    def V_g(self, x):
        self.Vgcount += 1
        return self.grad_log_prob(x)*-1.
       

    def H(self, q,p):
        self.Hcount += 1
        if self.returnV:
            V = self.V(q)
            return V, V + self.KE(p)
        else: return self.V(q) + self.KE(p)

    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        p = p - 0.5*step_size * self.V_g(q) 
        for i in range(N-1):
            q = q + step_size * self.KE_g(p)
            p = p - step_size * self.V_g(q) 
        q = q + step_size * self.KE_g(p)
        p = p - 0.5*step_size * self.V_g(q) 
        return q, p


    def metropolis(self, qp0, qp1):
        q0, p0 = qp0
        q1, p1 = qp1
        if self.returnV:
            V0, H0 = self.H(q0, p0)
            V1, H1 = self.H(q1, p1)
        else:
            H0 = self.H(q0, p0)
            H1 = self.H(q1, p1)
            V0, V1 = [None]*q0.shape[0], [None]*q0.shape[0]

        prob = np.exp(H0 - H1)
        acc  = []
        q2, p2, V2 = [], [], []
        for i in range(q0.shape[0]):
            iprob = prob[i]
            accept = 1
            if np.isnan(iprob) or (q0-q1).sum()==0: 
                accept = 2
            elif np.random.uniform(0., 1., size=1) > min(1., iprob):
                accept = 0
            acc.append(accept)
            if accept == 1:
                q0[i] = q1[i].copy()
                p0[i] = p1[i].copy()
                V0[i] = V1[i]                
        return q0, p0, np.array(acc), np.array([H0, H1, V0], dtype=np.float32)


#        for i in range(q0.shape[0]):
#            iprob = prob[i]
#            accept = 1
#            if np.isnan(iprob) or (q0-q1).sum()==0: 
#                accept = 2
#            elif np.random.uniform(0., 1., size=1) > min(1., iprob):
#                accept = 0
#            acc.append(accept)
#            print(i, iprob, accept)
#            if accept == 1:
#                q2.append(q1[i])
#                p2.append(p1[i])
#                V2.append(V1[i])
#            else: 
#                q2.append(q0[i])
#                p2.append(p0[i])
#                V2.append(V0[i])
#
#        return np.stack(q2, 0), np.stack(p2, 0), np.array(acc), np.array([H0, H1, V2], dtype=np.float32)
#

    def hmc_step(self, q, N, step_size):

        step_sizeb = step_size.copy()
        for idim in range(1, len(q.shape)): step_sizeb = np.expand_dims(step_sizeb, -1)
        p = np.random.normal(size=q.size).reshape(q.shape) #* self.metricstd
        q1, p1 = self.leapfrog(q, p, N, step_sizeb)
        q, p, accepted, energy = self.metropolis([q, p], [q1, p1])    
        return q, p, accepted, energy, [self.Hcount, self.Vgcount, self.leapcount]
