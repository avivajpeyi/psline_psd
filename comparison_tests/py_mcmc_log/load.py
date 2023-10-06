from slipper.sample.sampling_result import Result

r = Result.load("result.nc")
print(r)


import matplotlib.pyplot as plt

last_psd = r.psd_posterior[-1]
plt.plot(last_psd)
plt.show()
