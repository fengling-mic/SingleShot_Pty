import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # This makes GPU N appear as GPU 0 to CuPy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import h5py
import cupy as cp
import cupyx.scipy.fft
from scipy.ndimage import gaussian_filter

from ssPty.misc.misc import plot_hsv, guess_scalebar_size, recursively_save_dict_to_h5, print_h5_tree
from ssPty.RPI_sims.generate_rzp import generate_randomized_zone_plate_xray, make_fzp_trans_complex

num_gpus = cp.cuda.runtime.getDeviceCount()
for i in range(num_gpus):
    with cp.cuda.Device(i):  # Set context to this device
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        print(f"GPU {i}:\n  Free memory:  {free_mem / 1e6:.1f} MB\n  Total memory: {total_mem / 1e6:.1f} MB")


# change to your save path!
save_dir = 'Y:\\fengling\\RIP_simu_data\\'

# FZP geometry (um)
r_zp_um = 250./2 # zone plate outer radius
bs_zp_um = r_zp_um * 0.3 # beamstop inner radius
outer_width_zp_um = 50e-3 # 4 # outer zone width

# FZP
en_keV = 8.
wavelength_um = 1.24e-3 / en_keV
n_air = 1.
n_complex = (1 - 2.99120929e-5) - 2.2081831e-6j if abs(10 - en_keV) < 0.1 else (1 - 4.77303183e-5) - 4.96011535e-6j # 8 keV
thickness = 0.5 # Au
material = '%dAu' %(np.round(1e3*thickness))
# material = '30Cr'
# n_complex = 3.1414 + 1j * 3.3143 # Cr
# thickness = 30e-3 # Cr
# n_complex = (2.7021 + 1j * 3.7629, 0.18508 + 1j * 3.4233) # Ti, Au
# thickness = (5e-3, 30e-3) # Ti, Au
zp_trans_complex = make_fzp_trans_complex(wavelength_um, n_complex, thickness, n_air)

# FZP design
n_pillar = 10 # controls relative width of pillars, arb unit based on number of pillars in one circumference, smaller means larger pillars
pillar_frac_width = 0.4 # maximum 1 - duty_cycle of pillars
apodize_sigma = 1. # gaussian FWHM for apodization (units of r_zp_um - bs_zp_um)
probe_shape = 'square' # circ, square, tri

print('zone plate contrast: %.3f * exp(i %.2f)' %(np.abs(zp_trans_complex), np.angle(zp_trans_complex)))

# geometry (um)
ddet_um = 10e6
sensorPixSize_um = 172
zp_padding_frac = 1.3

nPix = 1.3*wavelength_um * ddet_um / sensorPixSize_um / outer_width_zp_um
nPix = int(np.round(nPix / 64)) * 64
res_obj = wavelength_um * ddet_um / (nPix * sensorPixSize_um)
Rmax_fac = 0.5  #beam shape vs FoV sample plane
Rmax = res_obj * nPix/2 * Rmax_fac

nPix_zp = int(np.ceil(zp_padding_frac*2*r_zp_um / res_obj / 256)) * 256

print('material:            %s' %material)
print('nPix in detector:    %d' %nPix)
print('nPix for ZP sim:     %d' %nPix_zp)
print('obj sample rate:     %.1f nm' %(res_obj*1e3))
print('focal length:        %.1f mm' %(1e-3*2 * r_zp_um * outer_width_zp_um / wavelength_um))
print('field of view:       %.2f um' %(wavelength_um * ddet_um / sensorPixSize_um))
print('Spot size:           %.2f um' %(2*Rmax))
print('lam = %.3f nm, ddet = %d mm, spix = %d um' %(wavelength_um*1e3, ddet_um*1e-3, sensorPixSize_um))

if 'fft_plan_zp_plane' not in globals():
    print('creating FFT plans')
    fft_plan_zp_plane = cupyx.scipy.fft.get_fft_plan(cp.zeros((nPix_zp, nPix_zp), dtype='complex64'))
    fft_plan_probe    = cupyx.scipy.fft.get_fft_plan(cp.zeros((nPix, nPix),       dtype='complex64'))

print('### generating initial probe ###')
out = generate_randomized_zone_plate_xray(r_zp_um, bs_zp_um, Rmax, outer_width_zp_um, nPix, nPix_zp, zp_trans_complex, res_obj, wavelength_um, ddet_um, fft_plan_zp_plane,
                                          fft_plan_probe, f_cutoff=1., print_fresnel=True, force_fresnel=False, n_pillar=n_pillar, pillar_frac_width=pillar_frac_width,
                                          apodize_sigma=apodize_sigma, plot_intermediate=False, probe_shape=probe_shape)

zp_true, zp_binary, probe_obj, probe_det, dzp, fft_plan_probe, res_zp = out
# initial_field, zp_true, zp_binary, probe_obj, probe_det, dzp, fft_plan_probe, res_zp = out
cp.get_default_memory_pool().free_all_blocks()

plt.figure(figsize=(7,7))
plt.title('binary zp, res %d nm' %(1e3*res_zp))
a, b = int(0.5*r_zp_um/np.sqrt(2)/res_zp) + nPix_zp//2, int(r_zp_um/res_zp)-1000
plt.imshow(cp.asnumpy((cp.abs(zp_binary) < 1)[a-300:a+20,a-300:a+20]), origin='lower')

plt.figure(figsize=(7,7))
plt.title('binary zp, res %d nm' %(1e3*res_zp))
a, b = int(r_zp_um/np.sqrt(2)/res_zp) + nPix_zp//2, int(r_zp_um/res_zp)-1000
plt.imshow(cp.asnumpy((cp.abs(zp_binary) < 1)[a-300:a+20,a-300:a+20]), origin='lower')

plt.figure(figsize=(14,8))
plt.subplot(121)
plt.title('binary zp, pixel size %d nm' %(1e3*res_zp))
plt.imshow((cp.abs(zp_binary) < 1).get(), origin='lower', extent=[-nPix_zp*res_zp*1e-3/2,nPix_zp*res_zp*1e-3/2,-nPix_zp*res_zp*1e-3/2,nPix_zp*res_zp*1e-3/2])
plt.xlim(-r_zp_um*1e-3,r_zp_um*1e-3)
plt.ylim(-r_zp_um*1e-3,r_zp_um*1e-3);
plt.axis('off')

sbar = 10e-3
scalebar = AnchoredSizeBar(plt.gca().transData, sbar, '%d um' %(1e3*sbar), 'lower right', pad=0.5, color='white', size_vertical=0.001, frameon=False)
plt.gca().add_artist(scalebar);

# plt.subplot(122)
# plt.title('binary zp, pixel size %d nm' %(1e3*res_zp))
# plt.imshow((cp.abs(zp_binary) < 1).get(), origin='lower', extent=[-nPix_zp*res_zp*1e-3/2,nPix_zp*res_zp*1e-3/2,-nPix_zp*res_zp*1e-3/2,nPix_zp*res_zp*1e-3/2])
# plt.xlim(0.6*r_zp_um*1e-3,0.75*r_zp_um*1e-3)
# plt.ylim(0.6*r_zp_um*1e-3,0.75*r_zp_um*1e-3);
# plt.axis('off')

# sbar = 1e-3
# scalebar = AnchoredSizeBar(plt.gca().transData, sbar, '%d um' %(1e3*sbar), 'upper right', pad=0.5, color='white', size_vertical=0.0001, frameon=False)
# plt.gca().add_artist(scalebar);

db = 4
xy = (3700, 3850)
# xy = (1500, 1850)
plt.figure(figsize=(18,5))
# plt.subplot(1,3,1)
# plt.title('initial field at focus, res %d nm' %(1e3*res_obj))
# plot_hsv(initial_field[-nPix//2+nPix_zp//2:nPix//2+nPix_zp//2,-nPix//2+nPix_zp//2:nPix//2+nPix_zp//2].get())
# sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(nPix, res_obj)
# scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
# plt.gca().add_artist(scalebar)

plt.subplot(1,3,2)
plt.title('initial field at zp, res %d nm' %(1e3*res_zp))
plot_hsv(zp_true[::db,::db])#.get())
sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(zp_true.shape[0], res_zp)
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)

plt.subplot(1,3,3)
plt.title('binary zp, res %d nm' %(1e3*res_zp))
a, b = int(r_zp_um/np.sqrt(2)/res_zp) + nPix_zp//2, int(r_zp_um/res_zp)
plot_hsv(cp.asnumpy(zp_binary[a-200:a+20,a-200:a+20]))
# plot_hsv(cp.asnumpy(zp_binary[nPix_zp//2:nPix_zp//2+b,nPix_zp//2:nPix_zp//2+b]))
sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(xy[1] - xy[0], res_zp)
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'upper right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)

plt.figure(figsize=(18,5))
plt.subplot(1,3,1)
plt.title('probe obj')
plot_hsv(cp.asnumpy(probe_obj))
sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(probe_obj.shape[0], res_obj)
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)

plt.subplot(1,3,2)
plt.title('probe det')
plot_hsv(cp.asnumpy(probe_det))
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)

plt.subplot(1,3,3)
plt.title('log(probe det)')
plt.imshow(np.log(cp.asnumpy(cp.abs(probe_det))), origin='lower', cmap='magma')
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)


print('### saving zone plate pattern ###')
zp_sim_design_fname = save_dir + 'h5_all/zp_sim_design_xray_d%dum_ow%dnm_px%dnm_%s_ampVar.h5' %(r_zp_um*2, 1e3*outer_width_zp_um, 1e3*res_zp, probe_shape)

zp_bool = cp.asnumpy(cp.angle(zp_binary)).astype(bool)

if not os.path.isfile(zp_sim_design_fname):
    zp_bool = cp.asnumpy(cp.angle(zp_binary)).astype(bool)

    to_save = {
        'data':{
            'zp_bool':zp_bool,
            'specific_sim_data':{
                material+'_complex_design':{
                    'probe_obj':probe_obj.get(),
                }
            },
        },
        'metadata':{
            'zp':{
                'r_zp':r_zp_um,
                'bs_zp':bs_zp_um,
                'outer_width_zp':outer_width_zp_um,
                'N_zp':nPix_zp,
                'res_zp':res_zp,
                'n_pillar':n_pillar,
                'pillar_frac_width':pillar_frac_width,
                'apodize_sigma':apodize_sigma,
                'probe_shape':np.array(probe_shape, dtype=h5py.string_dtype(encoding='utf-8')),
            },
            'detector':{
                'sensorPixSize':sensorPixSize_um,
                'N_probe':nPix,
            },
            'object_plane':{
                'res_obj':res_obj,
                'Rmax':Rmax,
            },
            'specific_sim_metadata':{
                material:{
                    'wavelength':wavelength_um,
                    'thickness':thickness,
                    'zp_trans_complex':zp_trans_complex,
                }
            }
        },
    }

    with h5py.File(zp_sim_design_fname, 'w') as f:
        recursively_save_dict_to_h5(f, "", to_save)
    print(zp_sim_design_fname)

else:
    with h5py.File(zp_sim_design_fname, 'r+') as f:
        if material not in f['/metadata/specific_sim_metadata'].keys():
            # add metadata for new material
            f['/metadata/specific_sim_metadata'].create_group(material)
            f['/metadata/specific_sim_metadata'][material]['wavelength'] = wavelength_um
            f['/metadata/specific_sim_metadata'][material]['thickness'] = thickness
            f['/metadata/specific_sim_metadata'][material]['zp_trans_complex'] = zp_trans_complex
            
            f['/data/specific_sim_data'].create_group(material+'_complex_design')
            f['/data/specific_sim_data'][material+'_complex_design']['probe_obj'] = probe_obj.get()
            print('file already exists:' + zp_sim_design_fname)
            print('adding new material:', material)

        else:
            print('file already exists: ' + zp_sim_design_fname)

print_h5_tree(zp_sim_design_fname)


zp_sim_design_fname = save_dir + 'h5_all/zp_sim_design_xray_d%dum_ow%dnm_px%dnm_%s_ampVar.h5' %(r_zp_um*2, 1e3*outer_width_zp_um, 1e3*res_zp, probe_shape)
with h5py.File(zp_sim_design_fname, 'r') as f:
    zp = f['/data/zp_bool'][()]

plt.figure()
plt.imshow(zp, origin='lower')

im_save_path = 'Y:\\fengling\\RIP_simu_data'

a = 3750, 3980
sbar_size = 200 * res_obj+1
plt.imshow(zp, origin='lower', cmap='gray');
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_size, '%d um' % sbar_size, 'lower right', pad=0.5, color='k', size_vertical=1, frameon=True)
# plt.gca().add_artist(scalebar)
plt.gca().set_axis_off()
plt.xlim(a)
plt.ylim(a);
# plt.savefig(im_save_path + 'rzp100um_bool.png')

def angular_spectrum_prop_vec_scaled(
    probe, z, res, wavelength, fft_plan_in,
    return_res_out=False
):
    N = probe.shape[0]

    # wavenumber
    k = cp.float32(2 * cp.pi / wavelength)
    z = cp.float32(z)

    # frequency grid
    k1 = (2 * cp.pi * cp.fft.fftshift(cp.fft.fftfreq(N, d=res))).astype(cp.float32)

    # FFT (black box)
    field = fft2_with_shift(probe, fft_plan_in)

    # Elementwise kernel with scaled k
    kernel = cp.ElementwiseKernel(
        in_params='complex64 f, raw float32 k1, int32 N, float32 k, float32 z',
        out_params='complex64 out',
        operation=r'''
            int iy = i / N;
            int ix = i - iy * N;

            // scale by k to reduce magnitude
            float kx_norm = k1[ix] / k;
            float ky_norm = k1[iy] / k;

            float kz2_norm = 1.0f - (kx_norm*kx_norm + ky_norm*ky_norm);

            // suppress evanescent waves
            kz2_norm = kz2_norm >= 0.0f ? kz2_norm : 0.0f;

            float phase = k * sqrtf(kz2_norm) * z;

            out = f * exp(complex<float>(0.0f, phase));
        ''',
        name='angular_spectrum_scaled'
    )

    kernel(field, k1, N, k, z, field)

    # IFFT back to spatial domain
    field = ifft2_with_shift(field, fft_plan_in)

    if return_res_out:
        return field, res
    else:
        return field


def angular_spectrum_prop_meshgrid(
    probe, z, res, wavelength, fft_plan_in,
    return_res_out=False, free_memory=False
):
    """
    Angular spectrum propagation (small Fresnel numbers)
    Meshgrid version with correct float64 for kz
    """
    N = probe.shape[0]

    # Critical quantities in float64
    k = cp.float64(2 * cp.pi / wavelength)
    z = cp.float64(z)
    k1 = (2 * cp.pi * cp.fft.fftshift(cp.fft.fftfreq(N, d=res))).astype(cp.float64)

    # Spatial frequency grids
    kx, ky = cp.meshgrid(k1, k1)

    # Longitudinal component (float64)
    kz2 = k*k - kx*kx - ky*ky
    kz = cp.sqrt(cp.where(kz2 >= 0, kz2, 0))  # float64

    if free_memory:
        del kx, ky, kz2
        cp.get_default_memory_pool().free_all_blocks()

    # FFT (probe stays complex64)
    field = fft2_with_shift(probe, fft_plan_in)

    # Multiply by phase factor (cast to complex64 after computation)
    phase = cp.exp(1j * (kz * z)).astype(cp.complex64)
    field *= phase
    del phase, kz
    if free_memory:
        cp.get_default_memory_pool().free_all_blocks()

    # IFFT back to spatial domain
    field = ifft2_with_shift(field, fft_plan_in).astype(cp.complex64)

    if return_res_out:
        return field, res
    else:
        return field


def angular_spectrum_prop2(probe, z, res, wavelength, fft_plan_in, return_res_out=False, free_memory=False):
    '''
    Angular spectrum propagation
    Used for small Fresnel numbers (< 1)
    '''
    
    N = probe.shape[0]
    k = 2 * cp.pi / wavelength

    # --- Spatial frequency grids ---
    k1 = 2 * cp.pi * cp.fft.fftshift(cp.fft.fftfreq(N, d=res))          # cycles per unit length
    k_reuse, ky = cp.meshgrid(k1, k1)

    # --- Longitudinal component ---
    k_reuse = k**2 - (k_reuse)**2 - (ky)**2
    if free_memory:
        del ky
        cp.get_default_memory_pool().free_all_blocks()

    k_reuse = cp.sqrt(cp.where(k_reuse >= 0, k_reuse, 0))  # suppress evanescent
    probe_out = ifft2_with_shift(cp.exp(cp.complex64(1j) * k_reuse * z).astype('complex64') * fft2_with_shift(probe, fft_plan_in), fft_plan_in).astype('complex64')

    if free_memory:
        del k_reuse
        cp.get_default_memory_pool().free_all_blocks()

    return (probe_out, res) if return_res_out else probe_out

# temp = angular_spectrum_prop_vec_scaled(
#     zp_true, 72.6e3, res_zp, wavelength_um, fft_plan_zp_plane,
#     return_res_out=False
# )
temp2 = angular_spectrum_prop2(
    zp_true, 72.6e3, res_zp, wavelength_um, fft_plan_zp_plane
)

plt.figure(figsize=(18,5))
# plt.subplot(1,3,1)
# plt.title('initial field at focus, res %d nm' %(1e3*res_obj))
# plot_hsv(temp[::8,::8].get())
# sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(nPix, res_obj)
# scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
# plt.gca().add_artist(scalebar)

plt.subplot(1,3,2)
plt.title('initial field at focus, res %d nm' %(1e3*res_obj))
plot_hsv(temp2[::8,::8].get())
sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(nPix, res_obj)
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)

plt.figure(figsize=(18,5))
# plt.subplot(1,3,1)
# plt.title('initial field at focus, res %d nm' %(1e3*res_obj))
# plot_hsv(temp[-nPix//2+nPix_zp//2:nPix//2+nPix_zp//2,-nPix//2+nPix_zp//2:nPix//2+nPix_zp//2].get())
# sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(nPix, res_obj)
# scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
# plt.gca().add_artist(scalebar)

plt.subplot(1,3,2)
plt.title('initial field at focus, res %d nm' %(1e3*res_obj))
plot_hsv(temp2[-nPix//2+nPix_zp//2:nPix//2+nPix_zp//2,-nPix//2+nPix_zp//2:nPix//2+nPix_zp//2].get())
sbar_size, sbar_pix_size, sbar_thickness, sbar_unit = guess_scalebar_size(nPix, res_obj)
scalebar = AnchoredSizeBar(plt.gca().transData, sbar_pix_size, '%d %s' %(sbar_size, sbar_unit), 'lower right', pad=0.5, color='white', size_vertical=sbar_thickness, frameon=False)
plt.gca().add_artist(scalebar)