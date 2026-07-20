import numpy as np
import cupy as cp
import cupyx.scipy.fft


#######################################################################
# 2D FFT wrapper for cupy
#######################################################################
def fft2_with_shift(arr_in, fft_plan_in):
    return cupyx.scipy.fft.fftshift(cupyx.scipy.fft.fft2(cupyx.scipy.fft.ifftshift(arr_in), plan=fft_plan_in))


def ifft2_with_shift(arr_in, fft_plan_in):
    return cupyx.scipy.fft.fftshift(cupyx.scipy.fft.ifft2(cupyx.scipy.fft.ifftshift(arr_in), plan=fft_plan_in))


#######################################################################
# 1D FFT wrapper for cupy
#######################################################################
def fft_with_shift(arr_in, fft_plan_in):
    return cupyx.scipy.fft.fftshift(cupyx.scipy.fft.fft(cupyx.scipy.fft.ifftshift(arr_in), plan=fft_plan_in))


def ifft_with_shift(arr_in, fft_plan_in):
    return cupyx.scipy.fft.fftshift(cupyx.scipy.fft.ifft(cupyx.scipy.fft.ifftshift(arr_in), plan=fft_plan_in))


#######################################################################
# 2D FFT wrapper for cupy with axes specified
#######################################################################
def fft2_with_shift_3d(arr_in, fft_plan_in, axes_in):
    return cupyx.scipy.fft.fftshift(cupyx.scipy.fft.fft2(cupyx.scipy.fft.ifftshift(arr_in, axes=axes_in), plan=fft_plan_in, axes=axes_in), axes=axes_in)


def ifft2_with_shift_3d(arr_in, fft_plan_in, axes_in):
    return cupyx.scipy.fft.fftshift(cupyx.scipy.fft.ifft2(cupyx.scipy.fft.ifftshift(arr_in, axes=axes_in), plan=fft_plan_in, axes=axes_in), axes=axes_in)

