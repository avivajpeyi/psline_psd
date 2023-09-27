import celerite2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from celerite2 import terms
from spectrum import Periodogram, arma2psd, aryule

from slipper.example_datasets import ar_data


def fmt_psd(raw_psd):
    return 10 * np.log10(abs(raw_psd) * 2.0 / (2.0 * np.pi))


# Create a AR model
a = [1, -2.2137, 2.9403, -2.1697, 0.9606]
# create some data based on these AR parameters
y = scipy.signal.lfilter([1], a, np.random.randn(1, 1024))
# if we know only the data, we estimate the PSD using Periodogram
p = Periodogram(y[0], sampling=2)  # y is a list of list hence the y[0]
data_psd = p.psd
# data_psd = data_psd[len(data_psd):len(data_psd) // 2:-1]
data_psd = fmt_psd(data_psd)
data_x = np.linspace(0, 1, len(data_psd))

# now, let us try to estimate the original AR parameters
AR, P, k = aryule(y[0], 4)
true_psd = arma2psd(AR, NFFT=512)
true_psd = true_psd[len(true_psd) : len(true_psd) // 2 : -1]
true_psd = fmt_psd(true_psd)
true_x = np.linspace(0, 1, len(true_psd))

plt.plot(data_x, data_psd, label="data")
plt.plot(true_x, true_psd, label="true")
plt.legend()

#
# pdgrm = ar_data.get_ar_periodogram(order=3, n_samples=5000)[2:-2]
# true_pdgrm =
# x = np.linspace(0, 1, len(pdgrm))
# plt.scatter(x, pdgrm, marker=',', color='k', s=1, alpha=0.25)
# plt.yscale('log')
# plt.show()
#
#
kernel = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)
gp = celerite2.GaussianProcess(kernel, mean=0.0)
gp.compute(data_x, yerr=0.5)


mu, variance = gp.predict(data_psd, t=true_x, return_var=True)
sigma = np.sqrt(variance)
plt.plot(true_x, mu, label="prediction")
plt.fill_between(true_x, mu - sigma, mu + sigma, color="C0", alpha=0.2)

plt.show()

# gp.compute(pdgrm.data, yerr=pdgrm.data_y)


# print(gp.log_likelihood(pdgrm))
#
#
# mu, variance = gp.predict(y, t=true_t, return_var=True)
# sigma = np.sqrt(variance)
# plt.plot(true_t, mu, label="prediction")
# plt.fill_between(true_t, mu - sigma, mu + sigma, color="C0", alpha=0.2)
#
#
