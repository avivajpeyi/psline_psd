import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from slipper.sample.log_pspline_sampler.bayesian_functions import lprior
import numpy as np


# Load the R function
r_source = """
lprior <- function (k, v, tau, tau.alpha, tau.beta, phi, phi.alpha, phi.beta,
                delta, delta.alpha, delta.beta, P)
{
  # Sigma^(-1) = P

  logprior <- k * log(phi)/2 - phi * t(v) %*% P %*% v / 2 +  #MNormal on weights

  dgamma(phi, phi.alpha, delta * phi.beta, log = TRUE) +# log prior for phi

  dgamma(delta, delta.alpha, delta.beta, log = TRUE) + # log prior for delta

  dnorm(tau, 0, 100, log = TRUE); # prior for tau

  return(logprior)
}
"""
robjects.r(r_source)

# Input parameters
k = 3
v = robjects.FloatVector([1.2, 2.5])
tau = 1.0
tau_alpha = 0.5
tau_beta = 2.0
phi = 2.0
phi_alpha = 1.0
phi_beta = 0.5
delta = 0.1
delta_alpha = 1.0
delta_beta = 2.0
P = np.array([[0.1, 0.2], [0.3, 0.4]])
P = numpy2ri.numpy2rpy(P)

# Call lprior function
result = robjects.r['lprior'](k, v, tau, tau_alpha, tau_beta, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, P)


py_res = lprior(k, np.array(v), tau, phi, phi_alpha, phi_beta, delta, delta_alpha, delta_beta, np.array(P))

# Print the result
print(result)
print(py_res)