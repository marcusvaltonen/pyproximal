# 0.5.0
* Added  ``pyproximal.proximal.Log1`` operator
* Allow ``radius`` parameter of ``pyproximal.optimization.primal.L0`` to be a function
* Allow ``tau`` parameter of ``pyproximal.optimization.primal.HQS`` to be a vector
  and change over iterations
* Added ``z0`` to ``pyproximal.optimization.primal.HQS``
* Added ``factorize`` option to ``densesolver`` of ``pyproximal.proximal.L2``

# 0.4.0
* Added ``pyproximal.optimization.primal.ADMML2``,
  `pyproximal.optimization.primal.HQS`,
  and ``pyproximal.optimization.pnp.PlugAndPlay`` solvers
* Added ``pyproximal.proximal.ETP``, ``pyproximal.proximal.Geman``,
  ``pyproximal.proximal.L0`, ``pyproximal.proximal.Log``,
  ``pyproximal.proximal.QuadraticEnvelopeCard``, ``pyproximal.proximal.SCAD``
  operators.
* Allow ``tau`` parameter of proximal operators to be a vector to handle problems with
  multiple right-hand sides.

# 0.3.0
* Added ``pyproximal.optimization.palm.PALM`` optimizer
* Added ``callback`` to ``pyproximal.optimization.proximal.ProximalPoint`` 
  optimizer
* Added ``pyproximal.utils.bilinear.BilinearOperator`` and 
  ``pyproximal.utils.bilinear.LowRankFactorizedMatrix`` operators

# 0.2.0
* Added ``pyproximal.proximal.L0Ball``, ``pyproximal.proximal.L1Ball``, 
  ``pyproximal.proximal.L21_plus_L1``, ``pyproximal.proximal.Nuclear``, 
  ``pyproximal.proximal.NuclearBall``, and ``pyproximal.proximal.Nonlinear`` 
  operators
* Added ``pyproximal.optimization.primal.TwIST`` solver
* Added `acceleration` in
  ``pyproximal.optimization.primal.AcceleratedProximalGradient`` solver
* Added classes standard deviation in
  ``pyproximal.optimization.segmentation.Segment`` solver
* Added `chain` method ``pyproximal.ProxOperator``
* Fix ``pyproximal.proximal.Orthogonal`` by introducing `alpha`
  in the proximal evaluation
  
# 0.1.0
* Added ``pyproximal.optimization.sr3.SR3`` solver
* Added ``pyproximal.projection.AffineSetProj`` and
  ``pyproximal.AffineSet`` operators
* Fixed ``pyproximal.Huber`` operator

# 0.0.0
* First official release.

