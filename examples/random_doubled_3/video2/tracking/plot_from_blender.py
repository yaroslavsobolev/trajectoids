import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('examples/random_doubled_3/video2/tracking/coordinates_from_blender.txt', delimiter='\t')
np.savetxt('examples/random_doubled_3/video2/tracking/trajectory_x.txt', data[:, 0])
np.savetxt('examples/random_doubled_3/video2/tracking/trajectory_y.txt', -1*data[:, 1])
plt.plot(data[:,0], data[:,1], 'o-')
plt.axis('equal')
plt.show()