import sirf.STIR
import cil.optimisation.algorithms as algorithms
from cil.optimisation.functions import Function, L2NormSquared

Class QuadraticPrior(Function):
    
    def __init__(self, beta, OSL = False, kappa=None, max_inner_iter = 5):
        
        if beta is not None:
            set_up(beta, OSL, kappa)
            
    def set_up(self, beta, OSL, kappa, max_inner_iter) 
        self.beta = beta # prior weighting
        self.kappa = kappa # anisotropic pixel weighting image - not existing as yet
        self.OSL = OSL
        
        self.iter = max_inner_iter
        
        self.prior = sirf.STIR.QuadraticPrior()
        self.prior.set_penalisation_factor(beta)
        
    def __call__(self, image):
        return self.prior.value(image)
    
    def gradient(self, image):
        return self.prior.get_gradient(image)
    
    def proximal(self,image):
        if OSL is True:
            raise NotImplementedError("Not yet exposed")
            return self.prior.get_proximal
        else:
            g = L2NormSquared(b=image)
            algorithm = FISTA(max_iteration=self.iter)
            algorithm.set_up(initial = image, f=self.prior, g=g)
            algorithm.run(verbose=1)
            return algorithm.solution     
        
    def set_penalisation_facor(value):
        self.prior.set_penalisation_factor(value)
        
        
        