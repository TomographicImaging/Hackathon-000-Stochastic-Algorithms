from sirf.STIR import OSMAPOSLReconstructor
import sirf.STIR as pet

from cil.optimisation.functions import L2NormSquared, IndicatorBox, LeastSquares, BlockFunction
from cil.optimisation.algorithms import Algorithm
from cil.processors import Slicer
from NewSubsetSumFunction import SGDFunction
from cil.plugins.astra.operators import ProjectionOperator
from NewFISTA import ISTA

### TODO --- COMMENTS!! ###
### CIL smoothing ###

class WarmStart(Algorithm):
    
    r''' function producing PET/CT image for the warm start of an algorithm
    
    uses OSEM with the option of post smoothign
    Parameters:
    -----------
    - Acquisition Model for data
    '''
    
    def __init__(self, initial = None, acquired_data=None, acq_model=None, modality=None, step_size = None, smooth=True, fwhms = (4,4,4), num_subsets=2, num_iters = 10, use_gpu = False, **kwargs):
        
        self.modality = modality
        self.step_size = step_size
        self.num_subsets = num_subsets
        self.num_iters = num_iters
        self.smooth = smooth
        self.fwhms = fwhms
        
        if use_gpu == True:
            self.device = 'gpu'
        else:
            self.device = 'cpu'
        
        if acquired_data is not None and initial is not None:
            set_up(acquired_data, acq_model, initial)
    
    def set_up(self, acquired_data, acq_model, initial):
        self.acquired_data = acquired_data
        
        self.initial = initial
        self.x = self.initial.clone()
        
        if acq_model is not None:
            self.acq_model = acq_model
        else:
            self.new_acq_model()
                
        if self.modality is None:
            raise ValueError("Please select modality ('PET'/'CT')")
        elif self.modality == 'PET':
            self.obj_fun = pet.make_Poisson_loglikelihood(self.acquired_data)
            self.obj_fun.set_acquisition_model(self.acq_model)
            self.reconstructor = OSMAPOSLReconstructor()
            self.reconstructor.set_objective_function(self.obj_fun)
            self.reconstructor.set_num_subsets(self.num_subsets)
            self.reconstructor.set_num_subiterations(self.num_iters*self.num_subsets)
            self.reconstructor.set_up(self.x)
            
        elif self.modality == 'CT':
            f_subsets = []
            for i, Ai in enumerate(self.acq_model):
                # Define F_i and put into list
                data_subset = Slicer(roi = {'angle' : (i,self.num_angles,self.num_subsets)})(self.acquired_data)
                fi = LeastSquares(Ai, b = data_subset)
                f_subsets.append(fi)
            F = BlockFunction(*f_subsets)
            self.obj_fun = SGDFunction(F)
            self.reconstructor = ISTA(initial=initial, f=self.obj_fun, g= IndicatorBox(0),
                        step_size=self.step_size, update_objective_interval=self.num_subsets, 
                        max_iteration=self.num_iters*self.num_subsets)
            
    
    def new_acq_model(self):
        if self.modality is None:
            raise ValueError("modality ('PET'/'CT') or acquisition model must be specified")
        elif self.modality == 'PET':
            if self.device == 'gpu':
                self.acq_model = pet.AcquisitionModelUsingParallelProj()
            else:
                self.acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
            self.acq_model.set_up(self.acquired_data)
        elif self.modality == 'CT':
            self.acq_model = []
            for i in range(self.num_subsets):
                # Total number of angles
                self.num_angles = len(self.acquired_data.geometry.angles)
                # Divide the data into subsets
                data_subset = Slicer(roi = {'angle' : (i,self.num_angles,self.num_subsets)})(self.acquired_data)

                # Define A_i and put into list 
                ageom_subset = data_subset.geometry
                Ai = ProjectionOperator(self.x.geometry, ageom_subset, device = self.device)
                self.acq_model.append(Ai)
        else:
            raise NotImplementedError()
    
    def run(self):
        if self.modality == 'PET':
            self.reconstructor.reconstruct(self.x)
        elif self.modality == 'CT':
            self.reconstructor.run(self.num_iters * self.num_subsets, verbose=0)
            self.x = self.reconstructor.solution
        if self.smooth is True:
            if self.modality == 'PET':
                smoother = pet.SeparableGaussianImageFilter()
                smoother.set_fwhms(self.fwhms)
                smoother.apply(self.x)            
    
    def update(self):
        if self.modality == 'PET':
            self.reconstructor.update()
        elif self.modality == 'CT':
            raise NotImplementedError()
        update_objective
    
    def update_objectvie(self):
        if self.modality == 'PET':
            return self.reconstructor.get_current_objective()
            
    
            
        