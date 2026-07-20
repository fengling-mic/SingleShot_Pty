import numpy as np
import cupy as cp

import sys
sys.path.append('/home/beams/ONEAL/code/Libraries/misc')
from fft_wrapper_cupy import fft2_with_shift, ifft2_with_shift

sys.path.append('/home/beams/ONEAL/code/Libraries/wave_propagation_cupy')
from wave_propagation_cupy_v2 import *


#######################################################################
# helper functions
#######################################################################

def get_fresnel_sample_rate(dz, d1, wv, N):
    return wv * abs(dz) / (d1 * N)


def make_fzp_trans_complex(wv, n_c, t, n_air):
    '''
    make complex transmission through FZP zones for binary patterning
    wv: wavelength
    n_c: complex index of refraction (single or list)
    t: material thickness (single or list, must match n_c)
    '''

    if (type(n_c) == type((1,))) or (type(n_c) == type([1,])):
        if (not ((type(t) == type((1,))) or (type(t) == type([1,])))):
            print('mismatched number of materials')
        if len(n_c) != len(t):
            print('mismatched number of materials')

        out = np.exp((-2 * (n_c[0] - n_air).imag + 1j * (n_c[0] - n_air).real) * np.pi * t[0] / wv)
        if len(n_c) == 1:
            return out
        
        for i in range(1, len(n_c)):
            out *= np.exp((-2 * (n_c[i] - n_air).imag + 1j * (n_c[i] - n_air).real) * np.pi * t[i] / wv)
        return out
    
    return np.exp((-2 * (n_c - n_air).imag + 1j * (n_c - n_air).real) * np.pi * t / wv)
            

#######################################################################
# Generate randomized zone plate for visible light experiment
#######################################################################

def generate_randomized_zone_plate_vis(r_zp, bs_zp, Rmax, outer_width_zp, N, N_zp, zp_trans_complex, res_foc, wavelength, sensorPixSize, fft_plan_zp_plane, fft_plan_probe, f_cutoff=1., print_fresnel=False, force_fresnel=False):
    """
    Generate BLR illumination pattern
    
    Parameters:
    kp (int): Maximum probe frequency in pixels
    upsampling (bool): Whether to upsample for object diffraction
    ko (int): Object frequency limit for upsampling (only needed if upsampling=True)
    
    Returns:
    numpy.ndarray: 2D array containing BLR illumination
    """

    # Setup array sizes
    x = cp.arange(-N_zp//2, N_zp//2, dtype=cp.float32)
    X, Y = cp.meshgrid(x, x)
    R = cp.sqrt(make_R(X, Y)) * res_foc
    del X, Y
    cp.get_default_memory_pool().free_all_blocks()

    foc = 2 * r_zp * outer_width_zp / wavelength
    print('zone plate focal length: %.1f mm' %(foc*1e-3))
    print('res at focus: %.1f nm' %(res_foc*1e3))
    
    # 1) Generate probe at focus
    initial_field = cp.zeros((N_zp, N_zp), dtype='complex64')
    circle_mask_foc = R <= Rmax
    initial_field[circle_mask_foc] = cp.exp(2j * cp.pi * cp.random.random(cp.sum(circle_mask_foc).get())).astype('complex64')
    initial_field_mean = initial_field.mean()

    # 2) Propagate to zone plate
    zp_true, res_zp = propagate_probe(initial_field, -foc, res_foc, wavelength, fft_plan_zp_plane, f_cutoff=f_cutoff, print_fresnel=print_fresnel, force_fresnel=force_fresnel, return_d2=True, free_memory=True)
    zp_true *= initial_field_mean / zp_true.mean()
    print('zp res: %.1f nm' %(res_zp*1e3))

    # 3) create binarized mask
    #       Create ring mask in Fourier space
    ring_mask = (R / res_foc * res_zp >= bs_zp) & (R / res_foc * res_zp <= r_zp)

    del R
    cp.get_default_memory_pool().free_all_blocks()

    zp_true *= ring_mask

    #       Create initial array and populate mask with ones
    zp_binary = cp.zeros_like(zp_true, dtype='complex64')

    #       Make binary mask
    zp_binary[ring_mask] = np.where(cp.angle(zp_true[ring_mask]) > 0., zp_trans_complex, 1.).astype('complex64')

    print(f"initial_field size: {initial_field.nbytes / (1024 ** 2):.2f} MB")
    print(f"zp_true size: {zp_true.nbytes / (1024 ** 2):.2f} MB")
    print(f"zp_binary size: {zp_binary.nbytes / (1024 ** 2):.2f} MB")
    print(f"ring_mask size: {ring_mask.nbytes / (1024 ** 2):.2f} MB")
    
    del ring_mask
    cp.get_default_memory_pool().free_all_blocks()

    # 4) propagate to object plane
    probe_obj = propagate_probe(zp_binary, foc, res_zp, wavelength, fft_plan_zp_plane, f_cutoff=f_cutoff, print_fresnel=print_fresnel, force_fresnel=force_fresnel, free_memory=True)

    # return initial_field, zp_true, zp_binary, probe_obj

    probe_obj = cp.ascontiguousarray(probe_obj[(N_zp-N)//2:(N_zp+N)//2, (N_zp-N)//2:(N_zp+N)//2]).astype('complex64')
    probe_obj *= initial_field_mean / probe_obj.mean()

    # 5) propagate to detector plane
    probe_det = fft2_with_shift(probe_obj, fft_plan_probe).astype('complex64')

    return initial_field, zp_true, zp_binary, probe_obj, probe_det, foc, fft_plan_probe, res_zp


#######################################################################
# Load zone plate design and generate probe
#######################################################################

def make_zp_binary_from_bool(zp_bool_in, zp_trans_complex, r_zp, bs_zp, res_zp):
    N_zp = zp_bool_in.shape[0]
    x = cp.arange(-N_zp//2, N_zp//2, dtype=cp.float32)
    X, Y = cp.meshgrid(x, x)
    R = make_R(X, Y) * res_zp**2
    del X, Y
    cp.get_default_memory_pool().free_all_blocks()

    ring_mask = (R >= bs_zp**2) & (R <= r_zp**2)
    zp_binary_out = cp.array(zp_bool_in, dtype='complex64')
    zp_binary_out[ring_mask] = cp.where(zp_bool_in[ring_mask], zp_trans_complex, 1.).astype('complex64')
    
    del ring_mask
    cp.get_default_memory_pool().free_all_blocks()

    return zp_binary_out


def make_probe_from_zp(zp_binary, res_zp, r_zp, outer_width_zp, wavelength, N, fft_plan_zp_plane, fft_plan_probe, f_cutoff=1., print_fresnel=False, force_fresnel=False):
    foc = 2 * r_zp * outer_width_zp / wavelength
    N_zp = zp_binary.shape[0]

    # 1) propagate to object plane
    probe_obj = propagate_probe(zp_binary, foc, res_zp, wavelength, fft_plan_zp_plane, f_cutoff=f_cutoff, print_fresnel=print_fresnel, force_fresnel=force_fresnel, free_memory=True)

    # return initial_field, zp_true, zp_binary, probe_obj

    probe_obj = cp.ascontiguousarray(probe_obj[(N_zp-N)//2:(N_zp+N)//2, (N_zp-N)//2:(N_zp+N)//2]).astype('complex64')
    probe_obj *= cp.mean(cp.abs(zp_binary)) / cp.mean(cp.abs(probe_obj))

    # 2) propagate to detector plane
    probe_det = fft2_with_shift(probe_obj, fft_plan_probe).astype('complex64')

    return probe_obj, probe_det