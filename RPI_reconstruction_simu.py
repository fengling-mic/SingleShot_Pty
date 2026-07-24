#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # This makes GPU N appear as GPU 0 to CuPy

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import h5py

import cupy as cp
import cupyx.scipy.fft
import cupyx.scipy.ndimage as cpx_nd

#%% import self-developed package
from ssPty.misc.misc import make_hsv, plot_hsv, guess_scalebar_size
from ssPty.misc.image_processing import register_translation
from ssPty.RPI_sims.setup_rzp_sims_vis import create_initial_object
from ssPty.fourier_shell_corr.fourier_shell_corr import fourier_shell_corr
from ssPty.wave_propagation_cupy.fourier_padding_cupy import fourier_pad, fourier_crop
from ssPty.misc.fft_wrapper_cupy import fft2_with_shift, ifft2_with_shift

num_gpus = cp.cuda.runtime.getDeviceCount()
for i in range(num_gpus):
    with cp.cuda.Device(i):  # Set context to this device
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        print(f"GPU {i}:\n  Free memory:  {free_mem / 1e6:.1f} MB\n  Total memory: {total_mem / 1e6:.1f} MB")


def get_fresnel_sample_rate(dz, d1, wv, N):
    return wv * abs(dz) / (d1 * N)

use_simple = True

# FZP geometry (um)
r_zp = 10000
bs_zp = r_zp * 0.3
outer_width_zp = 12
# res_zp = 2.3 # 5
wavelength = 0.8
sensorPixSize = 5.48
ddet = 10e6
nPix = 1024
res_obj = get_fresnel_sample_rate(ddet, sensorPixSize, wavelength, nPix)
res_zp = res_obj
N_obj_diff = 2048

material = '1000Au'
material_key = material+'_chunked_design' if use_simple else material+'_complex_design'
diff_exp_key = 'diffraction_experiment_%dpix' %N_obj_diff

h = 6.62607015e-34  # Planck's constant in J*s
c = 299792458  # Speed of light in m/s
J_eV = 1.60217663e-19  # Elementary charge in C
lambda_to_eV = lambda lam_in: (h * c) / (lam_in * J_eV)

# data_path = 'Y:\\fengling\\RIP_simu_data\\'
data_path = '/mnt/micdata3/fengling/RPI_sim/'
zp_sim_design_fname = data_path + 'h5_all/zp_sim_design_xray_d%dum_ow%dnm_px%dnm.h5' %(r_zp*2, outer_width_zp*1e3, res_zp*1e3)
zp_sim_diff_patterns_fname = data_path + 'h5_all/zp_sim_diff_patterns_d%dum_ow%dnm_px%dnm_%s_%ddiffpix.h5' %(r_zp*2, outer_width_zp*1e3, res_zp*1e3, material_key, N_obj_diff)

with h5py.File(zp_sim_design_fname, 'r') as f:
    sensorPixSize, N_probe = f['/metadata/detector/sensorPixSize'][()], f['/metadata/detector/N_probe'][()]
    N_zp, res_obj = f['/metadata/zp/N_zp'][()], f['/metadata/object_plane/res_obj'][()]

    print(f['/data/specific_sim_data/%s' %material_key].keys())

    exp_data_key = '/data/specific_sim_data/%s/%s' %(material_key, diff_exp_key)
    pos, pos_diff = f[exp_data_key]['pos'][()], f[exp_data_key]['pos_diff'][()]
    probe_gt = f['/data/specific_sim_data/%s' %material_key]['probe_obj'][()]
    diff_pattern_gt = f[exp_data_key]['diff_pattern_gt'][()]

    exp_metadata_key = '/metadata/specific_sim_metadata/%s/%s' %(material, diff_exp_key)
    N_ke, N_obj, N_obj_db = f[exp_metadata_key]['N_ke'][()], f[exp_metadata_key]['N_obj'][()], f[exp_metadata_key]['N_obj_db'][()]
    idx_cen, n_tile, zoom = f[exp_metadata_key]['idx_cen'][()], f[exp_metadata_key]['n_tile'][()], f[exp_metadata_key]['zoom'][()]
print(zp_sim_design_fname)

with h5py.File(zp_sim_diff_patterns_fname, 'r') as f:
    diff_pattern_rpi = cp.array(f['diff_patterns'][idx_cen])
    probe_ptycho, obj_ptycho = f['/recon/obj/probe'][()], f['/recon/obj/object'][()]
print(zp_sim_diff_patterns_fname)


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(np.abs(obj_ptycho), cmap='gray', origin='lower')
# plt.scatter(pos[:,0].get(), pos[:,1].get(), c='r', s=2)
plt.subplot(1,3,2)
plt.imshow(np.angle(obj_ptycho), cmap='viridis', origin='lower')
plt.title('recon_object')
plt.subplot(1,3,3)
plot_hsv(probe_ptycho[0])
plt.title('recon_probe')
plt.show()

# ePIE parameters
alpha = 0.5
beta = 0.1
n_iter = 2001
update_max_freq = 10 # number of iterations between probe_max and obj_max updates
probe_update_delay = 10 # number of iterations before updating probe
plot_freq = 200 # number of iterations between plots

db = 4
obj_shape = [384] * 2
print(obj_shape)

obj = cp.ones(obj_shape, dtype='complex64') # initial object
recon_probe_rpi = cp.array(probe_ptycho[0]) # initial probe

if 'fft_plan_obj' in globals():
    del fft_plan_obj
fft_plan_obj = cupyx.scipy.fft.get_fft_plan(obj)

if 'fft_plan_probe' in globals():
    del fft_plan_probe
fft_plan_probe = cupyx.scipy.fft.get_fft_plan(recon_probe_rpi)

Pprefix = alpha * (cp.conj(recon_probe_rpi) / cp.max(np.abs(recon_probe_rpi))**2)
err_abs, err_gt = np.zeros(n_iter//5+1, dtype='float32'), np.zeros(n_iter//5+1, dtype='float32')
diff_pattern_sq = cp.sqrt(cp.array(diff_pattern_rpi))

for i in range(n_iter):
    # pad object
    obj_pad = fourier_pad(obj, (N_probe, N_probe), fft_plan_obj, fft_plan_probe)

    # create exit wave
    exitWave = obj_pad * recon_probe_rpi
    exitWave_fft = fft2_with_shift(exitWave, fft_plan_probe)

    # calculate error
    if i % 5 == 0:
        err_abs[i//5] = cp.mean(cp.abs(cp.abs(exitWave_fft) - diff_pattern_sq))
        # err_gt[i//5]  = cp.mean(cp.abs(exitWave_fft - diff_pattern_gt))

    # update exit wave with diffraction pattern
    exitWave_fft_new = cp.abs(diff_pattern_sq) * cp.exp(1j*cp.angle(exitWave_fft))
    exitWave_new = ifft2_with_shift(exitWave_fft_new, fft_plan_probe)

    # update probe_max and obj_max
    if i % update_max_freq == 0:
        probe_max, obj_max = cp.max(np.abs(recon_probe_rpi))**2, cp.max(np.abs(obj))**2

    # update probe and object
    obj += fourier_crop(alpha * (cp.conj(recon_probe_rpi) / probe_max) * (exitWave_new - exitWave), obj_shape, fft_plan_probe, fft_plan_obj)
    if i > probe_update_delay:
        recon_probe_rpi += beta * (cp.conj(fourier_pad(obj, (N_probe, N_probe), fft_plan_obj, fft_plan_probe)) / obj_max) * (exitWave_new - exitWave)

    # plot
    if i % plot_freq == 0:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        hsv = make_hsv(cp.asnumpy(obj))
        plt.imshow(matplotlib.colors.hsv_to_rgb(hsv), origin='lower')
        plt.title('Object iteration %d' % i)
        plt.subplot(1, 4, 2)
        hsv = make_hsv(cp.asnumpy(recon_probe_rpi))
        plt.imshow(matplotlib.colors.hsv_to_rgb(hsv), origin='lower')
        plt.title('Probe')
        plt.subplot(1, 4, 3)
        hsv = make_hsv(cp.asnumpy(exitWave_new[::db,::db]))
        plt.imshow(matplotlib.colors.hsv_to_rgb(hsv), origin='lower')
        plt.title('Exit wave new')
        plt.subplot(1, 4, 4)
        hsv = make_hsv(cp.asnumpy(exitWave_fft_new))
        plt.imshow(matplotlib.colors.hsv_to_rgb(hsv), origin='lower')
        plt.title('Exit wave FFT new')
        plt.show()

# plot results
plt.figure(figsize=(15,2))
plt.subplot(1,2,1)
plt.semilogy(np.arange(0, n_iter, 5), err_abs);
plt.xlabel('Iteration')
plt.ylabel('amplitude error')
# plt.subplot(1,2,2)
# plt.semilogy(np.arange(0, n_iter, 5), err_gt);
# plt.xlabel('Iteration')
# plt.ylabel('ground truth error')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cp.asnumpy(cp.abs(obj)), cmap='gray', origin='lower')
plt.colorbar()
plt.title('Object iteration %d' % i)
plt.subplot(1, 2, 2)
plt.imshow(cp.asnumpy(cp.angle(obj)), origin='lower')
plt.colorbar();

# if 'im_zoom' not in globals():
im_path = '/home/beams/ONEAL/data/'
N_obj_recon = obj_shape[0]
zoom = N_obj_recon/1024
n_tile = 2
im, im_zoom = create_initial_object(im_path, zoom=zoom, im_size_choice='large', n_tile=n_tile)
N_im, N_zoom = im.shape[0], im_zoom.shape[0]
posi = pos[idx_cen] - im.shape[0]//2
im_crop = im[N_im//2-N_probe//2+posi[0]:N_im//2+N_probe//2+posi[0],
             N_im//2-N_probe//2+posi[1]:N_im//2+N_probe//2+posi[1]].get()

posi_zoom = (posi * N_zoom // N_im).astype(int)
print(posi_zoom)
im_zoom_crop = im_zoom[N_zoom//2-N_obj_recon//2+posi_zoom[0]:N_zoom//2+N_obj_recon//2+posi_zoom[0],
                       N_zoom//2-N_obj_recon//2+posi_zoom[1]:N_zoom//2+N_obj_recon//2+posi_zoom[1]].get()


plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(cp.asnumpy(cp.abs(obj)), cmap='gray', origin='lower', vmax=20)
plt.colorbar()
plt.title('Object iteration %d' % i)
plt.subplot(2, 2, 2)
plt.imshow(cp.asnumpy(cp.angle(obj)), origin='lower', cmap='viridis')
plt.colorbar();
plt.subplot(2, 2, 3)
plt.imshow(np.abs(im_zoom_crop), origin='lower', cmap='gray')
plt.colorbar();
plt.subplot(2, 2, 4)
plt.imshow(np.angle(im_zoom_crop), origin='lower', cmap='viridis')
plt.colorbar();

# temp = median_filter(np.abs(obj.get()), 3) * np.exp(1.j * median_filter(np.angle(obj.get()), 3))
temp = obj.get()
im_zoom_crop_shift, shift_val = register_translation(temp, im_zoom_crop, use_phase_only=True)
print(shift_val)

FSC, T, C = fourier_shell_corr(im_zoom_crop_shift, temp, SNRt=0.5, alpha=0.5, to_print=False)
plt.title('Crossing: %.3f' %((np.where(FSC[2:T.size] - T[2:] < 0)[0][0] + 2) / T.size))
FSC, T, C = fourier_shell_corr(im_zoom_crop_shift, temp, SNRt=0.5, alpha=0.5)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(np.abs(temp), cmap='gray', origin='lower', vmax=20)
plt.colorbar()
plt.title('Object iteration %d' % i)
plt.subplot(2, 2, 2)
plt.imshow(np.angle(temp), origin='lower', cmap='viridis')
plt.colorbar();

plt.subplot(2, 2, 3)
plt.imshow(np.abs(im_zoom_crop_shift) * np.abs(temp).mean() / np.abs(im_zoom_crop_shift).mean() - np.abs(temp), origin='lower', cmap='RdBu_r')
plt.clim(-5, 5)
plt.colorbar();
plt.subplot(2, 2, 4)
plt.imshow(np.angle(im_zoom_crop_shift) - np.angle(temp), origin='lower', cmap='RdBu_r')
plt.clim(-np.pi, np.pi)
plt.colorbar();

def RPI_ePIE_recon(diff_pattern, probe_initial, im_in, fft_plan_probe, obj_shape_in, n_iter, alpha, beta, update_max_freq, probe_update_delay):
    obj = cp.ones(obj_shape_in, dtype='complex64') # initial object

    fft_plan_obj = cupyx.scipy.fft.get_fft_plan(obj)

    probe_recon = cp.copy(probe_initial) # initial probe
    diff_pattern_sq = cp.sqrt(diff_pattern)

    for i in range(n_iter):
        # pad object
        obj_pad = fourier_pad(obj, (N_probe, N_probe), fft_plan_obj, fft_plan_probe)

        # create exit wave
        exitWave = obj_pad * probe_recon
        exitWave_fft = fft2_with_shift(exitWave, fft_plan_probe)

        # update exit wave with diffraction pattern
        exitWave_fft_new = cp.abs(diff_pattern_sq) * cp.exp(1j*cp.angle(exitWave_fft))
        exitWave_new = ifft2_with_shift(exitWave_fft_new, fft_plan_probe)

        # update probe_max and obj_max
        if i % update_max_freq == 0:
            probe_max, obj_max = cp.max(np.abs(probe_recon))**2, cp.max(np.abs(obj))**2

        # update probe and object
        obj += fourier_crop(alpha * (cp.conj(probe_recon) / probe_max) * (exitWave_new - exitWave), obj_shape_in, fft_plan_probe, fft_plan_obj)
        if i > probe_update_delay:
            probe_recon += beta * (cp.conj(fourier_pad(obj, (N_probe, N_probe), fft_plan_obj, fft_plan_probe)) / obj_max) * (exitWave_new - exitWave)

    del fft_plan_obj
    err_diff_pattern = cp.mean(cp.abs(cp.abs(exitWave_fft) - diff_pattern_sq))
    
    im_crop_shift, shift_val = register_translation(obj.get(), im_in.get(), use_phase_only=True)

    temp = np.sqrt(np.sum(np.abs(im_crop_shift)**2))
    temp_fac = np.sum(im_crop_shift * cp.conj(obj).get()) / cp.sum(cp.abs(obj)**2).get()
    err_obj = np.sqrt(np.sum(np.abs(im_crop_shift - temp_fac * obj.get())**2)) / temp

    FSC, T, C = fourier_shell_corr(im_crop_shift, obj.get(), SNRt=0.5, alpha=0.5, to_print=False)
    FSC_crossing = (np.where(FSC[4:T.size] - T[4:] < 0)[0][0] + 4) / T.size
    
    return obj, probe_recon, err_diff_pattern, err_obj, FSC_crossing, im_crop_shift

# ePIE parameters
alpha = 0.5
beta = 0.01
n_iter = 2001
update_max_freq = 10 # number of iterations between probe_max and obj_max updates
probe_update_delay = 100 # number of iterations before updating probe
plot_freq = 200 # number of iterations between plots


obj_shape_list = np.array([192, 256, 320, 384, 416, 448, 480, 512, 768])
# obj_shape_list = np.array([64, 128, 320, 512])

obj_list, im_crop_shift = [], []
err_diff_pattern = np.empty_like(obj_shape_list)
err_obj = np.empty_like(obj_shape_list, dtype=float)
FSC_crossing = np.empty_like(obj_shape_list, dtype=float)

for i in range(obj_shape_list.size):
    print('%d / %d' %(i+1, obj_shape_list.size))
    obj_shape = [obj_shape_list[i]] * 2

    # make appropriately sized im for error calculation
    im_crop_temp_zoom = cpx_nd.zoom(cp.array(im_crop), zoom=obj_shape[0] / N_probe, order=1)

    # run RPI w/ ePIE
    obj, probe_recon, err_diff_pattern[i], err_obj[i], FSC_crossing[i], im_temp = RPI_ePIE_recon(cp.array(diff_pattern_rpi), cp.array(probe_ptycho[0]), im_crop_temp_zoom, fft_plan_probe, obj_shape, n_iter, alpha, beta, update_max_freq, probe_update_delay)
    obj_list.append(obj)
    im_crop_shift.append(im_temp)


colors = plt.cm.coolwarm(np.linspace(0, 1, obj_shape_list.size))
for i in range(obj_shape_list.size):
    FSC, T, C = fourier_shell_corr(im_crop_shift[i], obj_list[i].get(), SNRt=0.5,alpha=0.5, to_print=False)
    xx = np.linspace(0, 1 / (1024 / obj_shape_list[i] * res_obj), T.size)
    # xx = np.linspace(0, 1, T.size)
    if i == 0:
        plt.plot(xx, T, linestyle='--', c='k', label='1 bit threshold')
    else:
        plt.plot(xx, T, linestyle='--', c='k')
    plt.plot(xx, FSC[:T.size], c=colors[i], label='%d pix (%.1f um/pix, %.1f crossing)' %(
        obj_shape_list[i], 1024 / obj_shape_list[i] * res_obj, (1024 / obj_shape_list[i] * res_obj) / FSC_crossing[i]))
plt.legend();
plt.title('Fourier ring correlation w/ ground truth')
plt.xlabel('Spatial frequency (arb)');