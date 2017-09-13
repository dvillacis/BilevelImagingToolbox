import bilevel_imaging_toolbox as bit
import numpy as np
import matplotlib.pyplot as plt
import time

def _blocksignal():
    """ Generate a blocky signal for the demo """
    N = 1000
    s = np.zeros((N,1))
    s[int(N/4):int(N/2)] = 1
    s[int(N/2):int(3*N/4)] = -1
    s[int(3*N/4):int(-N/8)] = 2
    return s

### ROF Filtering
# Generate a blocky signal
s = _blocksignal()

# Add noise
n = s + 0.5*np.random.rand(*np.shape(s))

# Filter using the ROF model using a Proximal Newton Algorithm
lam = 20
print('Filtering signal using the ROF Model ...')
start = time.time()
f = bit.rof_1d(n,lam,method='forward-backward')
end = time.time()
print('Elapsed time ' + str(end-start))

# Plot Results
#plt.subplot(3, 1, 1)
#plt.title('ROF filtering')
#plt.plot(s)
#plt.ylabel('Original')
#plt.grid(True)

#plt.subplot(3, 1, 2)
#plt.plot(n)
#plt.ylabel('Noisy')
#plt.grid(True)

#plt.subplot(3, 1, 3)
#plt.plot(f)
#plt.ylabel('Filtered')
#plt.grid(True)

#plt.show()




