import numpy as np

from ..distributions import (ExponentialFamily,
                            MultivariateGaussian,
                            UnivariateGaussian,
                            VonMises,
                            IndGaussVM,
                             )

def _num_free_params_for_component(comp: ExponentialFamily) -> int:
    """Return the number of free parameters for a single component.

    We count independent degrees of freedom (e.g., full symmetric covariance has d(d+1)/2,
    not d^2). This is used for AIC/BIC.
    """
    if isinstance(comp, MultivariateGaussian):
        d = comp.d
        return d + (d * (d + 1)) // 2
    if isinstance(comp, UnivariateGaussian):
        return 2
    if isinstance(comp, VonMises):
        return 2
    if isinstance(comp, IndGaussVM):
        # composite: count the underlying parts
        return _num_free_params_for_component(comp.gaussian) + _num_free_params_for_component(comp.vonmises)

    # Fallback: best-effort count
    params = comp.params
    if not isinstance(params, (tuple, list)):
        params = (params,)
    count = 0
    for p in params:
        arr = np.asarray(p)
        if arr.ndim == 0:
            count += 1
        else:
            count += arr.size
    return count