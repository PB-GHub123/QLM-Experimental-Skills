import numpy as np
from matplotlib import pyplot as plt

spectrum = np.loadtxt('capture_3_michael.csv',delimiter = ',',dtype='float', skiprows=2)
spectrum.transpose()

plt.figure(1)
plt.plot(np.arange(spectrum.shape[0]), spectrum[:,1])
plt.show()