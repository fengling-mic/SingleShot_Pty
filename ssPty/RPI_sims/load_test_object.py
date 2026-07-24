import cupy as cp
import cupyx.scipy.fft
import cupyx.scipy.ndimage as cpx_nd
from PIL import Image


#######################################################################
# Load test object for ptychography simulations
#######################################################################

def load_test_object(im_path, zoom=0.5, im_size_choice='large', n_tile=1, base_size=1024):

    if im_size_choice == 'large':
        mag_in = Image.open(im_path + 'cameraman.tif').convert('F')
        ph_in = Image.open(im_path + 'mandril.tif').convert('F')
        # Normalize each source image (vertically flipped) to a base_size x base_size
        # object tile. The original code cropped larger bitmaps to 1024x1024; the
        # cameraman/mandril images are 512x512, so resample up to base_size to keep
        # the downstream invariants N_obj = base_size * n_tile and
        # im_for_diff = N_obj * zoom = N_obj_diff intact.
        mag_raw = cp.array(mag_in, dtype='complex64')[::-1]
        ph_raw  = cp.array(ph_in,  dtype='complex64')[::-1]
        mag = cpx_nd.zoom(mag_raw, (base_size / mag_raw.shape[0], base_size / mag_raw.shape[1]), order=1)
        ph  = cpx_nd.zoom(ph_raw,  (base_size / ph_raw.shape[0],  base_size / ph_raw.shape[1]),  order=1)
        if n_tile >= 2:
            n_roll = [mag.shape[0]//2]*2
            mag = cp.roll(cp.tile(mag, (n_tile, n_tile)), n_roll, axis=(0,1))
            ph  = cp.roll(cp.tile(ph,  (n_tile, n_tile)), n_roll, axis=(0,1))
    
        mag_zoom = cpx_nd.zoom(mag, zoom=zoom, order=1)
        ph_zoom = cpx_nd.zoom(ph, zoom=zoom, order=1)
    
    else:
        print("im size wrong: only choice is 'large'")

    im      = (mag      * cp.exp(1j*(ph      / cp.max(ph     )-1/2) * 2*cp.pi)).astype('complex64')
    im_zoom = (mag_zoom * cp.exp(1j*(ph_zoom / cp.max(ph_zoom)-1/2) * 2*cp.pi)).astype('complex64')

    print('im total size and zoom size:', im.shape, im_zoom.shape)
    return im, im_zoom