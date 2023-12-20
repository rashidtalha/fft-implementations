import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def fft_1d(arr):
    N = len(arr)
    if N == 1: return arr
    w = np.exp(2j * np.pi * np.arange(N//2) / N)
    ye, yo = fft_1d(arr[::2]), fft_1d(arr[1::2])
    y = np.zeros(N, dtype=np.complex128)
    m = N//2
    for k in range(m):
        y[k] = ye[k] + w[k] * yo[k]
        y[k + m] = ye[k] - w[k] * yo[k]
    return y

def fft_2d(arr):
    nrows, ncols = arr.shape
    res = np.zeros_like(arr, dtype=np.complex128)
    
    for row in range(nrows):
        res[row, :] = fft_1d(arr[row, :])

    for col in range(ncols):
        res[:, col] = fft_1d(res[:, col])

    return res

####################

# p = 4
# N = int(2**p)

# arr = np.ones((N, N))
# arr[:2,:2] = 0.5

# res = fft_2d(arr)
# res = res.real

# fig, ax = plt.subplots(ncols=2, figsize=(7,4), tight_layout=True)

# ax[0].imshow(arr)
# ax[1].imshow(res)

# ax[0].set_title("Original")
# ax[1].set_title("FFT 2D")

# plt.show()

####################

p = 10
N = int(2**p)

tmax = 10
t, delta = np.linspace(0, tmax, N, retstep=True)
y = np.sin(2*np.pi*3*t) + 0.85*np.sin(2*np.pi*4*t) - 0.15*np.sin(2*np.pi*7*t)

nyfreq = (N//2) * (1 / (delta * N))
f = np.linspace(0, nyfreq, N//2)

a = fft_1d(y)
a = a[:N//2]
a = np.abs(a)
# a = a.real

fig, ax = plt.subplots(ncols=2, figsize=(11,4), tight_layout=True)

ax[0].plot(t, y)
ax[0].set(xlim=(0, tmax))

ax[1].plot(f, a/np.max(a))
ax[1].set(xlim=(0, nyfreq))


plt.show()