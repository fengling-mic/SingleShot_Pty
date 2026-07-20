import numpy as np

import sys
sys.path.append('/home/beams/ONEAL/code/Libraries/misc')
from fft_wrapper import fft2_with_shift, ifft2_with_shift


def make_R(X_in, Y_in):
    R_out = np.empty_like(X_in)
    np.square(X_in, out=R_out)
    R_out += Y_in**2
    return R_out


#######################################################################
# Wave propagators
#######################################################################
def angular_spectrum_prop(probe, z, res, wavelength, return_res_out=False, free_memory=False):
    '''
    Angular spectrum propagation
    Used for small Fresnel numbers (< 1)
    '''
    
    N = probe.shape[0]
    k = 2 * np.pi / wavelength

    # --- Spatial frequency grids ---
    k1 = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=res))          # cycles per unit length
    kx, ky = np.meshgrid(k1, k1)

    # --- Longitudinal component ---
    kz = k**2 - kx**2 - ky**2
    if free_memory:
        del kx, ky
    kz = np.sqrt(np.where(kz >= 0, kz, 0))  # suppress evanescent
    probe_out = ifft2_with_shift(np.exp(np.complex64(1j) * kz * z).astype('complex64') * fft2_with_shift(probe)).astype('complex64')

    if free_memory:
        del kz

    return (probe_out, res) if return_res_out else probe_out
    

def fresnel_one_step(u_in, dz, d1, wv, return_d2=False, free_memory=False):
    """
    Fresnel numerical computation in one step.

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [um].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [um].
    dz : float
        Propagation distance [um].
    """
    N = u_in.shape[0]
    d2 = np.array(wv * np.abs(dz) / (N * d1), dtype=np.float32)
    k = np.array(2*np.pi / wv, dtype=np.float32)
    dz = np.array(dz, dtype=np.float32)
    d1 = np.array(d1, dtype=np.float32)

    x = np.arange(-N//2, N//2, dtype='float32')
    X, Y = np.meshgrid(x, x)
    R2 = make_R(X, Y)
    if free_memory:
        del X, Y

    u_mod = u_in * np.exp(np.complex64(1j) * k / (2 * dz) * (R2 * d1**2))

    if dz > 0:
        U = fft2_with_shift(u_mod)
    else:
        U = ifft2_with_shift(u_mod)
        
    quad_out = np.exp(np.complex64(1j) * (k * dz + k / (2 * dz) * (R2 * d2**2))) / (np.complex64(1j) * wv * dz)
    if free_memory:
        del R2

    return (quad_out * U, d2) if return_d2 else quad_out * U


def fresnel_two_step(u_in, dz, d1, d2, wv, return_d2=False):
    """
    Fresnel numerical computation that gives control over output sampling but at a higher cost of
    two FFTs.

    Parameters
    ----------
    u_in : :py:class:`~numpy.ndarray`
        Input amplitude distribution, [Ny, Nx].
    wv : float
        Wavelength [um].
    d1 : float
        Input sampling period for both x-dimension and y-dimension [um].
    d2 : float or list
        Desired output sampling period for both x-dimension and y-dimension [um].
    dz : float
        Propagation distance [um].
    """
    if d1 == d2:
        raise ValueError("Cannot have d1=d2.")

    N = u_in.shape[0]
    print('N:', N)

    # magnification
    m = d2 / d1
    print('m:', m)

    # intermediate plane
    dz1 = dz / (1 - m)
    print('dz1:', dz1)

    u_itm = fresnel_one_step(u_in, dz, d1, wv, return_d2=False)
    d1a = wv * np.abs(dz1) / (N * d1)
    print('d1a:', d1a)

    # observation plane
    dz2 = dz - dz1
    print('dz2:', dz2)
    
    if return_d2:
        return fresnel_one_step(u_itm, dz2, d1a, wv), d2
    else:
        return fresnel_one_step(u_itm, dz2, d1a, wv)


#######################################################################
# Propagation wrapper
#######################################################################
def propagate_probe(u_in, dz, d1, wv, d2=None, f_cutoff=1, print_fresnel=False, return_d2=False, force_fresnel=False, free_memory=False):
    '''
    Choose propagator based on Fresnel number
    '''

    if np.abs(dz / wv) < 1e-3:
        if return_d2:
            return u_in, d2
        else:
            return u_in
    
    else:
        N = u_in.shape[0]
        Fresnel_number = np.abs((d1 * N/4)**2 / (wv * dz))

        if (Fresnel_number > f_cutoff) and (not force_fresnel):
            if print_fresnel: print('Fresnel number for distance %.2f um: %.2f\nAngular spectrum method' %(dz, Fresnel_number))
            return angular_spectrum_prop(u_in, dz, d1, wv, return_d2, free_memory)
        elif d2 is None:
            if print_fresnel: print('Fresnel number for distance %.2f um: %.2f\nFresnel one step' %(dz, Fresnel_number))
            return fresnel_one_step(u_in, dz, d1, wv, return_d2, free_memory)
        else:
            if print_fresnel: print('Fresnel number for distance %.2f um: %.2f\nFresnel two steps' %(dz, Fresnel_number))
            return fresnel_two_step(u_in, dz, d1, d2, wv, return_d2)
        



def propagate_probe_multiple(u_in, obj_z, dz, d1, wv, d2=None, f_cutoff=1, print_fresnel=False, return_d2=False, force_fresnel=False):
    '''
    Propagate probe multiple times
    At each step, multipy by optical element
    Choose propagator based on Fresnel number
    '''

    if return_d2:
        for i, z_i in enumerate(dz):
            u_in, d2_out = propagate_probe(u_in, z_i, d1, wv, d2, f_cutoff, print_fresnel, return_d2, force_fresnel)
            u_in *= obj_z[i]

        return u_in, d2_out
    
    else:
        for i, z_i in enumerate(dz):
            u_in = propagate_probe(u_in, z_i, d1, wv, d2, f_cutoff, print_fresnel, return_d2)
            u_in *= obj_z[i]

        return u_in



