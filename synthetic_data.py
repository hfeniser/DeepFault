
import numpy as np
import matplotlib.pyplot as plt

cov = [[2,0], [0,2]]

a_1 = np.random.multivariate_normal([1,1], cov, 100) # mu = (1,1)
a_2 = np.random.multivariate_normal([1,9], cov, 100) # mu = (1,9)
a_3 = np.random.multivariate_normal([5,5], cov, 100) # mu = (5,5)
a_4 = np.random.multivariate_normal([9,1], cov, 100) # mu = (9,1)
a_5 = np.random.multivariate_normal([9,9], cov, 100) # mu = (9,9)

plt.plot(a_1, 'x')
plt.axis('equal')
plt.show()

