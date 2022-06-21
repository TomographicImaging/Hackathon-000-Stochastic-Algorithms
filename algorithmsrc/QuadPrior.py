import sirf.STIR
from cil.optimisation.algorithms import GD
from cil.optimisation.functions import Function, L2NormSquared, SumScalarFunction

class Prior(Function):
    """ Prior Base Class """
    
    def __init__(self, prior, beta):
        self.prior = prior
        self.prior.set_penalisation_factor(beta)
        
    def __call__(self, image):
        return self.prior.value(image)
    
    def gradient(self, image):
        return self.prior.get_gradient(image)
    
    def set_up(self, image):
        self.prior.set_up(image)

class QuadraticPrior(Prior):
    
    def __init__(self, beta, tau = None, OSL = False, kappa=None, max_inner_iter = 5):
        
        if beta is not None:
            self.set_up(beta, tau, OSL, kappa, max_inner_iter)
            
    def set_up(self, beta, tau, OSL, kappa, max_inner_iter):
        self.beta = beta # prior weighting
        self.kappa = kappa # anisotropic pixel weighting image - not existing as yet
        self.OSL = OSL
        
        if tau is None:
            tau = 1
            print("warning - tau set to 1")
        self.tau = tau
        
        self.iter = max_inner_iter
        
        self.prior = Prior(prior = sirf.STIR.QuadraticPrior(), beta=self.beta)
    
    def proximal(self,image):
        self.prior.set_up(image)
        if self.OSL is True:
            raise NotImplementedError("Not yet exposed")
            return self.prior.get_OSL_proximal(image, tau)
        else:
            g = L2NormSquared(b=image)
            algorithm = GD(image, g+self.prior, max_iteration=self.iter)
            algorithm.run(verbose=1)
            return algorithm.solution     
        
    def set_penalisation_facor(value):
        self.prior.set_penalisation_factor(value)
        
        
        
