import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from ssPty.RPI_sims.fft_wrapper_cupy import fft2_with_shift, ifft2_with_shift
from ssPty.RPI_sims.wave_propagation_cupy import make_R, make_R_abs, make_R_tri, propagate_probe


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

        out = np.exp(((n_c[0] - n_air).imag + 1j * (n_c[0] - n_air).real) * 2 * np.pi * t[0] / wv)
        if len(n_c) == 1:
            return out
        
        for i in range(1, len(n_c)):
            out *= np.exp(((n_c[i] - n_air).imag + 1j * (n_c[i] - n_air).real) * 2 * np.pi * t[i] / wv)
        return out
    
    return np.exp(((n_c - n_air).imag + 1j * (n_c - n_air).real) * 2 * np.pi * t / wv)


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
    
    del R
    cp.get_default_memory_pool().free_all_blocks()
    
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
            

#######################################################################
# Generate randomized zone plate for x-ray experiment
#######################################################################

def generate_randomized_zone_plate_xray(r_zp, bs_zp, Rmax, outer_width_zp, N, N_zp, zp_trans_complex, res_foc, wavelength, ddet,
                                        fft_plan_zp_plane, fft_plan_probe, f_cutoff=1., print_fresnel=False, force_fresnel=False,
                                        n_pillar=400, pillar_frac_width=0.6, apodize_sigma=1., plot_intermediate=False, probe_shape='circ'):
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
        
    #       Create ring mask in Fourier space
    ring_mask = (R >= bs_zp) & (R <= r_zp)

    if probe_shape == 'circ':
        Rshp = cp.copy(R)
    elif probe_shape == 'square':
        Rshp = make_R_abs(X, Y) * res_foc
    elif probe_shape == 'square':
        Rshp = make_R_tri(X, Y) * res_foc
    else:
        print('probe_shape must be in (circ, square, tri)')

    X = X[ring_mask]
    Y = Y[ring_mask]

    R = R[ring_mask]

    # for making pillars that vary with wavefront amplitude
    if n_pillar > 0:
        Phi = (cp.arctan2(Y, X) + cp.pi)

    # apodization
    if apodize_sigma > 0:
        sigma = (r_zp - bs_zp) / cp.sqrt(cp.log(2.0))
        A = cp.exp(-((cp.clip((R - r_zp) / (r_zp - bs_zp), 0, 1) - 0.5) / sigma)**2)

    del X, Y
    cp.get_default_memory_pool().free_all_blocks()

    foc = 2 * r_zp * outer_width_zp / wavelength
    print('zone plate focal length: %.1f mm' %(foc*1e-3))
    print('res at focus: %.1f nm' %(res_foc*1e3))
    
    # 1) Generate probe at focus
    circle_mask_foc = Rshp <= Rmax
    del Rshp
    cp.get_default_memory_pool().free_all_blocks()

    zp_true = cp.zeros((N_zp, N_zp), dtype='complex64')
    zp_true[circle_mask_foc] = cp.exp(2.j * cp.pi * cp.random.random(cp.sum(circle_mask_foc).get())).astype('complex64')
    del circle_mask_foc
    cp.get_default_memory_pool().free_all_blocks()

    # 2) Propagate to zone plate
    zp_true, res_zp = propagate_probe(zp_true, -foc, res_foc, wavelength, fft_plan_zp_plane, f_cutoff=f_cutoff, print_fresnel=print_fresnel, force_fresnel=force_fresnel, return_d2=True, free_memory=True)
    print('zp res: %.1f nm' %(res_zp*1e3))
    if res_zp != res_foc: print('!!! zp and focus res not the same, this code will not work !!!!')

    # 3) create binarized mask

    # if plot_intermediate:
        # plt.figure()
        # plt.title('A')
        # plt.imshow(A.get(), origin='lower')
        # plt.figure()
        # plt.title('G')
        # plt.imshow(Phi[1600:1700,1600:1700].get(), origin='lower')
        # plt.figure()
        # plt.title('angle')
        # plt.imshow(cp.angle(zp_true[1600:1700,1600:1700]).get(), origin='lower')
        # plt.figure()
        # plt.title('ZP')
        # plt.imshow((cp.pi*R[1600:1700,1600:1700]**2/r_zp/outer_width_zp).get(), origin='lower')
        # plt.figure()
        # plt.title('N')
        # plt.imshow(((0 + (cp.pi*R**2/r_zp/outer_width_zp/2 + cp.angle(zp_true) + cp.pi)/2))[1600:1700,1600:1700].get(), origin='lower', vmin=150, vmax=170)
        # plt.colorbar()
        # plt.figure()
        # # plt.plot(((cp.pi*R**2/r_zp/outer_width_zp/2 + cp.angle(zp_true) + cp.pi)/2)[N_zp//2,1600:1700].get())
        # plt.plot((cp.round((cp.pi*R**2/r_zp/outer_width_zp/2 + cp.angle(zp_true) + cp.pi)/cp.pi, 0))[N_zp//2,1600:1700].get())
        # plt.figure()
        # plt.title('G+N')
        # # plt.imshow(((Phi + (cp.pi*R**2/r_zp/outer_width_zp/2 + cp.angle(zp_true) + cp.pi)/2))[1600:1700,1600:1700].get() % (2*np.pi), origin='lower')
        # plt.imshow(((0.*Phi + (Phi / r_zp**2 + 0.) * (cp.pi*R**2/r_zp/outer_width_zp/2 + cp.angle(zp_true) + cp.pi)/2))[800:1700,800:1700].get() % (2*np.pi), origin='lower')

    if n_pillar > 0:
        N_var = cp.round((cp.pi*R**2/r_zp/outer_width_zp/2 + cp.angle(zp_true[ring_mask]) + cp.pi) / cp.pi, 0)
        phi0_per_zone = 2 * cp.pi * cp.random.rand(int(N_var.max() + 1))

        Phi = (Phi * cp.sqrt(N_var * r_zp * outer_width_zp * 2)*n_pillar)
        Phi = (Phi + phi0_per_zone[N_var.astype(int)]) % (2*cp.pi)
        # Phi = (Phi[ring_mask] * n_pillar**2 + N_var*1) % (2*cp.pi)

    #       Apply apodization
    zp_true[~ring_mask] = 0
    if apodize_sigma > 0:
        zp_true[ring_mask] *= A
        del R, A
    else:
        del R
    if n_pillar > 0: del N_var
    cp.get_default_memory_pool().free_all_blocks()

    #       Create initial array and populate mask with ones
    zp_binary = cp.zeros_like(zp_true, dtype='complex64')

    #       Make binary mask, maybe using amplitude to determine pillar width
    if n_pillar > 0:
        zp_binary[ring_mask] = cp.where((cp.angle(zp_true[ring_mask]) > 0.
                                        ) & (2.*cp.pi*cp.abs(zp_true[ring_mask]) / cp.max(cp.abs(zp_true[ring_mask])) * pillar_frac_width + (0.9 - pillar_frac_width)*2*cp.pi > Phi
                                            ), zp_trans_complex, 1.).astype('complex64')
    else:
        zp_binary[ring_mask] = cp.where(cp.angle(zp_true[ring_mask]) > 0., zp_trans_complex, 1.).astype('complex64')

    if n_pillar > 0: del Phi
    cp.get_default_memory_pool().free_all_blocks()

    print(f"zp_true size: {zp_true.nbytes / (1024 ** 2):.2f} MB")
    print(f"zp_binary size: {zp_binary.nbytes / (1024 ** 2):.2f} MB")
    print(f"ring_mask size: {ring_mask.nbytes / (1024 ** 2):.2f} MB")
    zp_true_cpu = zp_true.get()
    
    del ring_mask, zp_true
    cp.get_default_memory_pool().free_all_blocks()

    # 4) propagate to object plane
    probe_obj = propagate_probe(zp_binary, foc, res_zp, wavelength, fft_plan_zp_plane, f_cutoff=f_cutoff, print_fresnel=print_fresnel, force_fresnel=force_fresnel, free_memory=True)
    if plot_intermediate:
        plt.figure()
        plt.title('probe_obj full field')
        temp = gaussian_filter(cp.log(cp.abs(probe_obj[(N_zp-4*N)//2:(N_zp+4*N)//2, (N_zp-4*N)//2:(N_zp+4*N)//2])).get(), 3)
        
        plt.imshow(temp / temp.max(), origin='lower', vmax=1, vmin=-1)
        plt.colorbar();

    probe_obj = cp.ascontiguousarray(probe_obj[(N_zp-N)//2:(N_zp+N)//2, (N_zp-N)//2:(N_zp+N)//2]).astype('complex64')
    probe_det = propagate_probe(probe_obj, ddet, res_foc, wavelength, fft_plan_probe, f_cutoff=f_cutoff, print_fresnel=print_fresnel, force_fresnel=force_fresnel).astype('complex64')

    return zp_true_cpu, zp_binary, probe_obj, probe_det, foc, fft_plan_probe, res_zp