from scipy.signal import butter, lfilter, freqz
import numpy as np

def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def inv_DFT_slow(x):
    x = np.asarray(x, dtype=np.complex64)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return np.dot(M, x)/N

class Preprocessor():
    def __init__(self, length = 3000, *args, **kwargs) -> None:
        self.THRESHOLD = 0.98/2
        self.LENGTH = length
        self.mask = [1 if abs(self.LENGTH//2-i) > self.LENGTH*self.THRESHOLD else 0 for i in range(self.LENGTH)]
        self.mask = np.array(self.mask,dtype=np.float32)
        pass

    def process_slow(self, arr):
        arr = DFT_slow(arr)
        N = len(arr)
        arr = [x if abs(N//2-i) > N*self.THRESHOLD else 0 for i,x in enumerate(arr)]
        arr[0] = 0
        arr = np.real(inv_DFT_slow(arr))
        return arr
    def process_fast(self, arr):
        arr = np.fft.fft(arr)
        arr = arr*self.mask
        arr[0] = 0
        arr = np.real(np.fft.ifft(arr))
        return arr