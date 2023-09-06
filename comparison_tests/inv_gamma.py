import numpy as np
from bilby.core.prior import Gamma, Prior
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaincinv, gammaln, xlogy



class InverseGamma(Prior):
    def __init__(self, alpha, beta, name=None, latex_label=None, unit=None, boundary=None):
        super(InverseGamma, self).__init__(name=name, minimum=0., latex_label=latex_label,
                                           unit=unit, boundary=boundary)
        self.alpha = alpha
        self.beta = beta

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate Gamma prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        return gammaincinv(self.k, val) * self.theta

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ==========
        val:  Union[float, int, array_like]

        Returns
        =======
         Union[float, array_like]: Prior probability of val
        """
        return np.exp(self.ln_prob(val))

    def ln_prob(self, val):
        """Returns the log prior probability of val.

        Parameters
        ==========
        val: Union[float, int, array_like]

        Returns
        =======
        Union[float, array_like]: Prior probability of val
        """
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _ln_prob = -np.inf
            else:
                _ln_prob = xlogy(self.k - 1, val) - val / self.theta - xlogy(self.k, self.theta) - gammaln(self.k)
        else:
            _ln_prob = -np.inf * np.ones(val.size)
            idx = (val >= self.minimum)
            _ln_prob[idx] = xlogy(self.k - 1, val[idx]) - val[idx] / self.theta \
                            - xlogy(self.k, self.theta) - gammaln(self.k)
        return _ln_prob

    def cdf(self, val):
        if isinstance(val, (float, int)):
            if val < self.minimum:
                _cdf = 0.
            else:
                _cdf = gammainc(self.k, val / self.theta)
        else:
            _cdf = np.zeros(val.size)
            _cdf[val >= self.minimum] = gammainc(self.k, val[val >= self.minimum] / self.theta)
        return _cdf


gamma_samples = Gamma(k=3, theta=2).sample(100000)
inv_gamma_samples = 1 / gamma_samples
dir_inv_gamma_samples = 1 / InverseGamma(k=3, theta=1 / 2).sample(100000)
fig, ax = plt.subplots(2, 1)
bins = np.geomspace(0.01, 5, 50)
ax[0].hist(inv_gamma_samples, bins=bins, density=True, label="Inverse Gamma", histtype="step")
ax[0].hist(dir_inv_gamma_samples, bins=bins, density=True, label="Direct Inverse Gamma", histtype="step")

ax[0].set_xscale("log")
ax[0].legend()
ax[1].hist(gamma_samples, bins=50, density=True, label="Gamma", histtype="step")
plt.legend()
plt.show()
