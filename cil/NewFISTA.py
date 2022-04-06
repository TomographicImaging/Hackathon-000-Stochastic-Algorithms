# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi
#   (Collaborative Computational Project in Tomographic Imaging), with
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.algorithms import Algorithm
import numpy
import numpy as np
import warnings
from numbers import Number

class FISTA(Algorithm):

    r"""Fast Iterative Shrinkage-Thresholding Algorithm, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
    .. math::
        \begin{cases}
            x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))\\
            t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
            y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k-1}}(x_{k} - x_{k-1})
        \end{cases}
    is used to solve
    .. math:: \min_{x} f(x) + g(x)
    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha^{k}`
    is the :code:`step_size` per iteration.
    Parameters
    ----------
    initial : DataContainer
            Starting point of the algorithm
    f : Function
        Differentiable function
    g : Function
        Convex function with *simple* proximal operator
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of FISTA.
                The default :code:`step_size` is :math:`\frac{1}{L}`.
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.
    Examples
    --------
    .. math:: \underset{x}{\mathrm{argmin}}\|A x - b\|^{2}_{2}
    >>> from cil.optimisation.algorithms import FISTA
    >>> import numpy as np
    >>> from cil.framework import VectorData
    >>> from cil.optimisation.operators import MatrixOperator
    >>> from cil.optimisation.functions import LeastSquares, ZeroFunction
    >>> np.random.seed(10)
    >>> n, m = 50, 500
    >>> A = np.random.uniform(0,1, (m, n)).astype('float32') # (numpy array)
    >>> b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32') # (numpy vector)
    >>> Aop = MatrixOperator(A) # (CIL operator)
    >>> bop = VectorData(b) # (CIL VectorData)
    >>> f = LeastSquares(Aop, b=bop, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> fista = FISTA(initial = ig.allocate(), f = f, g = g, max_iteration=10)
    >>> fista.run()
    See also
    --------
    :class:`.FISTA`
    :class:`.GD`
    """

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, val):
        if isinstance(val, Number):
            if val<=0:
                raise ValueError("Positive step size is required. Got {}".format(val))
            self._step_size = val
        else:
            raise ValueError("Step size is not a number. Got {}".format(val))

    def _set_step_size(self, step_size):

        """Set the default step size
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self.step_size = 1./self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self.step_size = step_size

    @property
    def convergence_criterion(self):
        return self.step_size > 1./self.f.L

    def _check_convergence_criterion(self):
        """Check convergence criterion
        """
        if isinstance(self.f.L, Number):
            if self.convergence_criterion:
                warnings.warn("Convergence criterion is not satisfied.")
                return False
            return True
        else:
            raise ValueError("Function f is not differentiable")

    def __init__(self, initial, f, g, step_size = None, **kwargs):

        super(FISTA, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))

        self._step_size = None

        # set up FISTA
        self.set_up(initial=initial, f=f, g=g, step_size=step_size, **kwargs)

    def set_up(self, initial, f, g, step_size, **kwargs):
        """ Set up of the algorithm
        """

        self.initial = initial
        self.f = f
        self.g = g

        # set step_size
        self._set_step_size(step_size=step_size)

        # check convergence criterion for FISTA is satisfied
        if kwargs.get('check_convergence_criterion', True):
            self._check_convergence_criterion()

        print("{} with {} setting up".format(self.__class__.__name__, self.f.__class__.__name__))

        # Initialise iterates, the gradient estimator, and the temporary variables
        self.x_old = initial.clone()
        self.x = initial.clone()

        self.gradient_estimator = self.x * 0.0
        self.tmp2 = self.x * 0.0
        self.tmp1 = self.x * 0.0

        self.configured = True
        print("{} with {} configured".format(self.__class__.__name__, self.f.__class__.__name__))


    def update(self):

        r"""Performs a single iteration of FISTA
        .. math::
            \begin{cases}
                x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))\\
                t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
                y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k-1}}(x_{k} - x_{k-1})
            \end{cases}
        """

        self.t_old = self.t
        self.f.gradient(self.y, out=self.u)
        self.u *= -self.step_size
        self.u += self.y

        self.g.proximal(self.u, self.step_size, out=self.x)

        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))

        self.x.subtract(self.x_old, out=self.y)
        self.y.axpby(((self.t_old-1)/self.t), 1, self.x, out=self.y)

        self.x_old.fill(self.x)


    def update_objective(self):
        """ Updates the objective
        .. math:: f(x) + g(x)
        """
        self.loss.append( self.f(self.x) + self.g(self.x) )

class ISTA(FISTA):

    r"""Iterative Shrinkage-Thresholding Algorithm, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.
    Iterative Shrinkage-Thresholding Algorithm (ISTA)
    .. math:: x^{k+1} = \mathrm{prox}_{\alpha^{k} g}(x^{k} - \alpha^{k}\nabla f(x^{k}))
    is used to solve
    .. math:: \min_{x} f(x) + g(x)
    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha^{k}`
    is the :code:`step_size` per iteration.
    Note
    ----
    For a constant step size, i.e., :math:`a^{k}=a` for :math:`k\geq1`, convergence of ISTA
    is guaranteed if
    .. math:: \alpha\in(0, \frac{2}{L}),
    where :math:`L` is the Lipschitz constant of :math:`f`, see :cite:`CombettesValerie`.
    Parameters
    ----------
    initial : DataContainer
              Initial guess of ISTA.
    f : Function
        Differentiable function
    g : Function
        Convex function with *simple* proximal operator
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of ISTA.
                The default :code:`step_size` is :math:`\frac{0.99 * 2}{L}.`
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.
    Examples
    --------
    .. math:: \underset{x}{\mathrm{argmin}}\|A x - b\|^{2}_{2}
    >>> from cil.optimisation.algorithms import ISTA
    >>> import numpy as np
    >>> from cil.framework import VectorData
    >>> from cil.optimisation.operators import MatrixOperator
    >>> from cil.optimisation.functions import LeastSquares, ZeroFunction
    >>> np.random.seed(10)
    >>> n, m = 50, 500
    >>> A = np.random.uniform(0,1, (m, n)).astype('float32') # (numpy array)
    >>> b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32') # (numpy vector)
    >>> Aop = MatrixOperator(A) # (CIL operator)
    >>> bop = VectorData(b) # (CIL VectorData)
    >>> f = LeastSquares(Aop, b=bop, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> ista = ISTA(initial = ig.allocate(), f = f, g = g, max_iteration=10)
    >>> ista.run()
    See also
    --------
    :class:`.FISTA`
    :class:`.GD`
    """

    @property
    def convergence_criterion(self):
        return self.step_size > 0.99*2.0/self.f.L

    def __init__(self, initial, f, g, step_size = None, **kwargs):

        super(ISTA, self).__init__(initial=initial, f=f, g=g, step_size=step_size, **kwargs)

    def _set_step_size(self, step_size):
        """ Set default step size.
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self.step_size = 0.99*2.0/self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self.step_size = step_size

    def update(self):

        r"""Performs a single iteration of ISTA
        .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))
        """

        # gradient step
        self.f.gradient(self.x_old, out=self.x)
        self.x *= -self.step_size
        self.x += self.x_old

        # proximal step
        # self.g.proximal(self.x, self.step_size, out=self.x)
        self.x = self.g.proximal(self.x, self.step_size)

        # update
        self.x_old.fill(self.x)


class AdaptiveMomentumISTA(Algorithm):
    """
        Momentum ISTA with iteration-dependent step-size and momentum
        'Accelerating variance-reduced stochastic gradient methods'
        Derek Driggs · Matthias J. Ehrhardt · Carola-Bibiane Schönlieb
        Mathematical Programming 2020
        Parameters
        ----------
        initial : DataContainer
                Starting point of the algorithm
        f : Function
            Differentiable function
        g : Function
            Convex function with *simple* proximal operator
        step_size : Function
            Function wich outputs the step-size at iteration k, gamma = step_size(k)
        momentum : Function
            Function wich outputs the momentum at iteration k, tau = momentum(k)
        kwargs: Keyword arguments
            Arguments from the base class :class:`.Algorithm`.
    """

    def  __init__(self, initial=None, f=None, g=None, step_size=None, momentum=None, **kwargs):

        super(AdaptiveMomentumISTA, self).__init__(**kwargs)

        self.set_up(initial=initial, f=f, g=g, step_size=step_size, momentum=momentum)


    def set_up(self, initial, f, g, step_size, momentum):

        self.f = f
        self.g = g
        self.x = initial.clone()
        self.grad_x = initial.clone()
        self.y = initial.clone()
        self.z = initial.clone()
        self.step_size = step_size
        self.momentum = momentum
        self.configured = True

    def update(self):
        '''Single iteration'''

        # get current step-size
        gamma = self.step_size(self.iteration)
        # get current momentum  
        tau = self.momentum(self.iteration)

        # apply momentum, store in x
        self.z.axpby(tau, 1-tau, self.y, out=self.x)
        # compute the gradient at x, store in grad_x
        self.f.gradient(self.x, out=self.grad_x)
        # take the gradient step, store in z
        self.z.axpby(1, -gamma, self.grad_x, out=self.z)
        # take the proximal step, store in z
        self.z = self.g.proximal(self.z, gamma)
        # apply momentum, store in y
        self.z.axpby(tau, 1-tau, self.y, out=self.y)

    def update_objective(self):
        """ Updates the objective
        .. math:: f(x) + g(x)
        """
        self.loss.append( self.f(self.x) + self.g(self.x) )


from numbers import Number

class KATYUSHA(Algorithm):
    r"""KATYUSHA algorithm for stochastic optimisation

    Reference
    Allen-Zhu, Zeyuan
    "Katyusha: The First Direct Acceleration of Stochastic Gradient Methods"
    Journal of Machine Learning Research, 18, pp 1-51, 2018
    https://www.jmlr.org/papers/volume18/16-410/16-410.pdf

    Used to solve
    .. math:: \min_x f(x) + g(x)
    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator

    This is a direct acceleration of the SVRG algorithm. See

    Parameters
    ----------

    initial : DataContainer
            Starting point of the algorithm
    f : Function
        Differentiable function
    g : Function
        Convex function with *simple* proximal operator
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.


    Examples
    --------
    """


    def _set_step_sizes(self, strong_convexity) :
        """
        Setting the stepsize and mixing parameters
        """
        from numbers import Number
        if isinstance(strong_convexity, Number):
            if strong_convexity <= 0: 
                raise ValueError("Strong convexity constant needs to be a positive real. Got {}".format(strong_convexity))
            else:
                # If the strong convexity constant is known then use values provided by theory
                self.strong_convexity = strong_convexity
                # this self.L might need to be self.f.L, or something like that
                self.tau1 = np.min([0.5, np.sqrt(self.inner_loop_length*self.strong_convexity/3./self.L)]) # 
                self.alpha = 1./3./self.tau1 /self.L
                self.weights = np.array([(1.0+self.tau1*self.alpha) ** k for k in range(self.inner_loop_length)])
                self.weight_normalisation = 1.0 / self.weights.sum()
        elif strong_convexity is None:
            # if the problem is not strongly convex or if strong convexity constant is not known
            self.strong_convexity = None
            self.tau1 = 0.5
            self.alpha = 1./(3.*self.tau1 * self.L)
            self.weights = np.ones(self.inner_loop_length)
            self.weight_normalisation = 1.0 / self.weights.sum()
        else:
            raise ValueError("Strong convexity constant needs to be a number. Got {}".format(strong_convexity))

    def _set_algorithm_option(self, option):
        if option is None:
            print("option not provided. Setting it to _first_")
            option = 'first'
        elif isinstance(option, str) is False:
            raise ValueError("option needs to be a string. first and second are admissible options. Got {}".format(option))
        elif option.lower() not in ['first', 'second']: 
            raise ValueError("option needs to be either first or second. Got {}".format(option))
        else:
            self.option = option


    def  __init__(self, initial=None, f=None, g=None, **kwargs):
        super(KATYUSHA, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))

        # Set up KATYUSHA
        self.set_up(initial=initial, f=f, g=g, ** kwargs)

    def set_up(self, initial, f, g, update_frequency = 2, strong_convexity = None, option = 'first', **kwargs):
        '''
        Set up the algorithm
        '''

        self.initial = initial
        self.f = f
        self.g = g

        #check that f is svrg and check that it's update frequency is inf
        if self.f.__class__.__name__ == 'SVRGFunction':
            if self.f.update_frequency is not np.inf:
                raise ValueError("SVRGFunction for katyusha needs to be set up with update frequency = np.inf. Got {}".format(self.f.update_frequency))
        else:
            raise ValueError("The block function f for KATYUSHA needs to be SVRGFunction. Got {}".format(self.f.__class__.__name__))

        # set uop the default parameters
        self.tau2 = 0.5
        self.L = self.f.Lmax 
        self.num_subsets = self.f.num_subsets
        self.inner_loop_length = update_frequency * self.num_subsets

        # define num_subsets and L
        print("{} setting up".format(self.__class__.__name__, ))
        self._set_step_sizes(strong_convexity)
        self._set_algorithm_option(option)
        
        # setting up the x, and the intermediate values z and y
        self.x = initial.clone()
        self.z = initial.clone()
        self.y = initial.clone()

        # self.snapshot = initial.clone() # initial full gradient should be computed at this # remove - should be done and stored by svrg
        self.snapshot_running_estimator = self.x * 0.0
        self.f.memory_init(initial)
        self.tmp1 = self.x * 0.0

        self.update_step_size = False        
        self.configured = True

        print("{} configured".format(self.__class__.__name__, ))        

    def update(self):
        r"""Performs a single iteration of KATYUSHA

        .. math::
            
            \text{For } s = 0,\ldots, S-1
            G^s = \nabla F({\tilde x}^s)
            \text{For } j = 0,\ldots, m-1
            \begin{cases}
                k = sm+j
                x_{k+1} = \tau_1 z_k + \tau_2 {\tilde x}^s + (1-\tau_1-\tau_2) y_k \\
                \tilde \Nabla_{k+1} = \nabla F_{i}(x_{k+1}) - \Nabla F_{i} ({\tilde x}^s)+G^s
                z_{k+1} = \argmin_{z} \Big(\frac{1}{2\alpha}\|z-z_{k}\|^2 + \langle \tilde \Nabla_{k+1}, z\rangle + g(z)\Big)
                        = \textrm{prox}_{\alpha g} (z_{k}-\alpha \Nabla_{k+1})
                First option:
                    y_{k+1} = \argmin_{y}  \Big(\frac{3L}{2}\|y-x_{k+1}\|^2 + \langle \tilde \Nabla_{k+1}, y\rangle + g(y)\Big)
                            = \textrm{prox}_{1/3L g} (x_{k+1} - 1/3L \tilde \Nabla_{k+1})
                Second option:
                    y_{k+1} = x_{k+1} + \tau_1(z_{k+1}-z_k)
            \end{cases}
            {\tilde x}^{s+1} = (\sum_{j=0}^{m-1} w_j)^{-1} (\sum_{j=0}^{m-1} w_j y_(sm+j+1))
        """

        '''Single iteration'''
        if self.iteration % self.inner_loop_length == 0 and self.iteration != 0:
            # update snapshot, the gradient at snapshot and reset the running snapshot
            # self.snapshot = self.snapshot_running_estimator.multiply(self.weight_normalisation) 
            # self.weight_normalisation * self.snapshot_running_estimator # is this some multiply with an out?
            self.f.memory_update(self.snapshot_running_estimator.multiply(self.weight_normalisation))
            # self.f.memory_update(self.snapshot)

            # in the non-strongly convex case tau1 and alpha are changed with respect to outer loop iterations
            if self.strong_convexity is None:
                self.tau1 = 2./(4. + self.iteration // self.inner_loop_length) # self.iteration // self.inner_loop_length tells how many complete inner loops have been done
                self.alpha = 1.0/3./self.tau1/self.L 
            self.snapshot_running_estimator = self.x * 0.0
        else:
            #compute x = tau1*z + tau2*snapshot +  (1-tau1-tau2)*y
            self.z.sapyb(self.tau1, self.f.snapshot, self.tau2, out = self.x)
            self.x.sapyb(1., self.y, 1-self.tau1-self.tau2, out = self.x)
            # self.x = self.tau1 * self.z + self.tau2 * self.f.snapshot + (1-self.tau1-self.tau2) * self.y
            # Compute the gradient direction with svrg
            self.f.gradient(self.x, out = self.tmp1)

            if self.option == 'first':
                # compute z - alpha*tmp1, where tmp1 = grad f(x)
                self.z.sapyb(1., self.tmp1, -self.alpha, out = self.z)              
                self.z = self.g.proximal(self.z, self.alpha)
                # compute x - 1/3/L * tmp1
                self.x.sapyb(1., self.tmp1, -1./3./self.L, out = self.tmp1)
                self.y = self.g.proximal(self.tmp1, 1./3./self.L)
                # self.y = self.g.proximal(self.x - 1./3./self.L * self.tmp1, 1./3./self.L)
            else: # if using the second option
                # temporarily store the current z into a dummy variable 
                self.z_tmp = self.z.clone()
                
                # compute z - alpha*tmp1, where tmp1 = grad f(x)
                self.z.sapyb(1., self.tmp1, -self.alpha, out = self.z)
                # compute proximal at z
                self.z = self.g.proximal(self.z, self.alpha)
                # self.z = self.g.proximal(self.z - self.alpha * self.tmp1, self.alpha)
                # Computing x + tau1 * (z-z_tmp)
                self.x.sapyb(1., self.z.subtract(self.z_tmp), self.tau1, out = self.y)
                # self.y = self.x + self.tau1 * (self.z - self.z_tmp)
            
            # instead of storing the num_inner_iterations values of y (needed to update the snapshot)
            # we compute a running estimator that is averaged out when the full gradient is updated,
            # giving the new snapshot. 
            # Multiply the current self.y, with a weight corresponding to the iteration of the inner loop
            # and add to the running estimator
            self.snapshot_running_estimator.sapyb(1., self.y, self.weights[self.iteration % self.inner_loop_length], out = self.snapshot_running_estimator) 
            # self.snapshot_running_estimator += self.weights[self.iteration % self.inner_loop_length] * self.y



    def update_objective(self):
        """ Updates the objective

        .. math:: f(x) + g(x)

        """
        self.loss.append( self.f(self.x) + self.g(self.x) )




class SARAH(Algorithm):

    r"""SARAH algorithm.

    StochAstic Recursive grAdient algoritHm (SARAH)
    Lam M. Nguyen, Jie Liu, Katya Scheinberg, Martin Takáč 
    Proceedings of the 34th International Conference on Machine Learning, PMLR 70:2613-2621, 2017. 
    https://proceedings.mlr.press/v70/nguyen17b/nguyen17b.pdf

    .. math::

        \begin{cases}
            
        \end{cases}

    It is used to solve

    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha^{k}`
    is the :code:`step_size` per iteration.


    Parameters
    ----------

    initial : DataContainer
            Starting point of the algorithm
    f : Function
        Differentiable function
    g : Function
        Convex function with *simple* proximal operator
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of SARAH
                The default :code:`step_size` is :math:`\frac{1}{L}`.
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.

    See also
    --------
    :class:`.FISTA`
    :class:`.GD`

    """

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, val):
        if isinstance(val, Number):
            if val<=0:
                raise ValueError("Positive step size is required. Got {}".format(val))
            self._step_size = val
        else:
            raise ValueError("Step size is not a number. Got {}".format(val))

    def _set_step_size(self, step_size):

        """Set the default step size
        """
        if step_size is None:
            if isinstance(self.L, Number):
                self.step_size = 1./self.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self.step_size = step_size

    def __init__(self, initial, f, g, step_size = None, update_frequency = 2, subset_select_function=(lambda a,b: int(np.random.choice(b))), subset_init=-1, precond = None, **kwargs):

        super(SARAH, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))

        self.subset_select_function = subset_select_function
        self.subset_num = subset_init

        self._step_size = None

        # set up SARAH
        self.set_up(initial=initial, f=f, g=g, step_size=step_size, update_frequency=update_frequency, precond=precond, **kwargs)

    def set_up(self, initial, f, g, step_size, update_frequency, precond, **kwargs): # update frequency
        """ Set up the algorithm
        """

        self.initial = initial
        self.f = f # at the moment this is required to be of SubsetSumFunctionClass (so that data_passes member exists)
        self.g = g

        # set problem parameters
        self.update_frequency = update_frequency
        self.precond = precond
        self.num_subsets = self.f.num_subsets
        self.L = self.f.Lmax
        self._set_step_size(step_size=step_size)
        # self.view_use = [] # should probably store this in f
        print("{} setting up".format(self.__class__.__name__, ))

        # Initialise iterates, the gradient estimator, and the temporary variables
        self.x_old = initial.clone()
        self.x = initial.clone()

        self.gradient_estimator = self.x * 0.0
        self.tmp2 = self.x * 0.0
        self.tmp1 = self.x * 0.0

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

    def update(self):

        r"""Performs a single iteration of SARAH

        .. math::
            # TODO: change maths
            \begin{cases}

            \end{cases}

        """

        self.gradient(self.x, out=self.gradient_estimator)
        self.x_old = self.x.clone()
        
        self.x.sapyb(1., self.gradient_estimator, -self.step_size, out = self.x)
        # self.x.axpby(1., -self.step_size, self.gradient_estimator, out = self.x)
        self.x = self.g.proximal(self.x, self.step_size) # not sure if this makes sense

    def gradient(self, x, out = None):            
        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()

        if self.iteration == 0 or self.iteration % (self.update_frequency * self.num_subsets) == 0: 
            self.f._full_gradient(self.x, out = self.gradient_estimator)
            # Update the number of (statistical) passes over the entire data until this iteration 
            self.f.data_passes.append(self.f.data_passes[-1]+1.)
            self.tmp2 = 0.0 * self.gradient_estimator
        else:
            # Compute new gradient for current subset, store in tmp1
            # tmp1 = gradient F_{subset_num} (x)
            self.f.functions[self.subset_num].gradient(x, out=self.tmp1) 
            # Update the number of (statistical) passes over the entire data until this iteration 
            self.f.data_passes.append(self.f.data_passes[-1]+1./self.num_subsets)

            # Compute difference between current subset function gradient at current iterate (tmp1) and at the previous iterate, store in tmp2
            # tmp2 = gradient F_{subset_num} (x) - gradient F_{subset_num} (x_old)
            self.tmp1.sapyb(1., self.f.functions[self.subset_num].gradient(self.x_old), -1., out=self.tmp2)
            # self.tmp1.axpby(1., -1., self.f.functions[self.subset_num].gradient(self.x_old), out=self.tmp2) 
        # Compute the output: tmp2 + full_grad
        if out is None:
            ret = 0.0 * self.tmp2
            self.tmp2.sapyb(1., self.gradient_estimator, 1., out=ret)
            # self.tmp2.axpby(1., 1., self.gradient_estimator, out=ret)
        else:
            self.tmp2.sapyb(1., self.gradient_estimator, 1., out=out)
            # self.tmp2.axpby(1., 1., self.gradient_estimator, out=out)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        if out is None:
            return ret

    def update_objective(self):
        """ Updates the objective

        .. math:: f(x) + g(x)

        """
        self.loss.append( self.f(self.x) + self.g(self.x) )

    def next_subset(self):
        self.subset_num = self.subset_select_function(self.subset_num, self.num_subsets)