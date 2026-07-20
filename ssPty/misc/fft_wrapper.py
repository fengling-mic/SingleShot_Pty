import numpy as np


#######################################################################
# 2D FFT wrapper
#######################################################################
def fft2_with_shift(arr_in):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr_in)))


def ifft2_with_shift(arr_in):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr_in)))


#######################################################################
# 1D FFT wrapper
#######################################################################
def fft_with_shift(arr_in):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(arr_in)))


def ifft_with_shift(arr_in):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(arr_in)))
