import numpy as np
from matplotlib import pyplot as plt

spectrum = np.loadtxt('capture_2_wider.csv',delimiter = ',',dtype='float', skiprows=2)
spectrum.transpose()

plt.figure(1)
plt.plot(spectrum[0], spectrum[2])
plt.show()