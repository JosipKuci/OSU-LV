import numpy as np
import matplotlib . pyplot as plt

zeroes=np.zeros((50,50))
ones=np.ones((50,50))
upper=np.hstack((zeroes,ones))
lower=np.hstack((ones,zeroes))
full=np.vstack((upper,lower))
full=full*255
print(full)
plt.figure ()
plt.imshow ( full ,cmap="gray")
plt.show ()
