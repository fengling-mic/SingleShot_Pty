import numpy as np
import cupy as cp

import sys
sys.path.append('/home/beams/ONEAL/code/Libraries/misc')
from fft_wrapper_cupy import fft2_with_shift, ifft2_with_shift


#######################################################################
# Wave propagation
#######################################################################
def angular_spectrum_prop(probe, z, res, wavelength, fft_plan_in, farfield_distance=1e7, return_res_out=False):
    '''
    Angular spectrum propagation
    Used for small Fresnel numbers (< 1)
    Set farfield_distance large enough to approximate plane wave
    '''

    N = probe.shape[0]
    pixel_size = wavelength * farfield_distance / (res * N)
    kx = cp.arange(-N//2, N//2, dtype='complex64') * pixel_size
    KX, KY = cp.meshgrid(kx, kx)
    opl_factor = -cp.pi * z / wavelength / farfield_distance**2

    probe_out = cp.exp(1j * opl_factor * (KX**2 + KY**2)) * fft2_with_shift(probe, fft_plan_in)
    probe_out = ifft2_with_shift(probe_out, fft_plan_in)

    if return_res_out:
        return probe_out, res
    else:
        return probe_out


def Fresnel_diffraction(probe, z, res, wavelength, fft_plan_in, return_res_out=False):
    '''
    Fresnel diffraction
    Used for large Fresnel numbers (> 1)
    '''

    N = probe.shape[0]
    x1 = cp.arange(-N//2, N//2, dtype='complex64') * res
    X1, Y1 = cp.meshgrid(x1, x1)
    opl1 = cp.pi * (X1**2 + Y1**2) / (wavelength * z)
    
    x2 = 2 * cp.pi * x1 / (wavelength * z)
    X2, Y2 = cp.meshgrid(x2, x2)
    opl2 = cp.pi * (X2**2 + Y2**2) / (wavelength * z)

    if return_res_out:
        res_out = wavelength * z / (res * N)
        if z > 0:
            return cp.exp(1j * (2 * cp.pi * z / wavelength * 0 + opl2)) * fft2_with_shift(probe * cp.exp(1j * opl1), fft_plan_in), res_out
        else:
            return cp.exp(1j * (2 * cp.pi * z / wavelength * 0 + opl2)) * ifft2_with_shift(probe * cp.exp(1j * opl1), fft_plan_in), res_out
        
    else:
        if z > 0:
            return cp.exp(1j * (2 * cp.pi * z / wavelength * 0 + opl2)) * fft2_with_shift(probe * cp.exp(1j * opl1), fft_plan_in)
        else:
            return cp.exp(1j * (2 * cp.pi * z / wavelength * 0 + opl2)) * ifft2_with_shift(probe * cp.exp(1j * opl1), fft_plan_in)


def propagate_probe(probe, z, res, wavelength, fft_plan_in, f_cutoff=1, farfield_distance=1e7, print_fresnel=False, return_res_out=False):
    '''
    Choose propagator based on Fresnel number
    Set farfield_distance large enough to approximate plane wave for angular spectrum propagation
    '''

    if np.abs(z / wavelength) < 1e-3:
        return probe
    
    else:
        N = probe.shape[0]
        Fresnel_number = np.abs((res * N/4)**2 / wavelength / z)
        if print_fresnel: print('Fresnel number for distance %.2f um: %.2f' %(z, Fresnel_number))

        if Fresnel_number > f_cutoff:
            return angular_spectrum_prop(probe, z, res, wavelength, fft_plan_in, farfield_distance, return_res_out)
        else:
            return Fresnel_diffraction(probe, z, res, wavelength, fft_plan_in, return_res_out)
    

def propagate_probe_multiple(probe, obj_z, z, res, wavelength, fft_plan_in=None, f_cutoff=1, farfield_distance=1e7):
    '''
    Propagate probe multiple times
    At each step, multipy by optical element
    Choose propagator based on Fresnel number
    Set farfield_distance large enough to approximate plane wave for angular spectrum propagation
    '''

    if fft_plan_in is None:
        fft_plan_in = cp.fft.get_fft_plan(probe)

    for i, z_i in enumerate(z):
        probe = propagate_probe(probe, z_i, res, wavelength, fft_plan_in, f_cutoff, farfield_distance)
        probe *= obj_z[i]

    return probe