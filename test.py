import random
import numpy as np
from matplotlib import pyplot as plt


a = [[random.randint(1, 4) for j in range(4)] for i in range(5)]
a = np.reshape(a, (2, 10))
print(a)

plt.scatter(a[0], a[1])
plt.show()



