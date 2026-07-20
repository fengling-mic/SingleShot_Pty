import cupy as cp
import cupyx.scipy.fft
import cupyx.scipy.ndimage as cpx_nd
from PIL import Image


#######################################################################
# Generate initial conditions for randomized zone plate visible light experiment
#######################################################################

def create_initial_object(im_path, zoom=384/1024, im_size_choice='large', n_tile=1):
    
    if im_size_choice == 'large':
        mag_in = Image.open(im_path + 'Rainier.bmp').convert('F')
        ph_in = Image.open(im_path + 'Malamute.bmp').convert('F')
        mag = cp.array(mag_in, dtype='complex64')[1023::-1,500:1024+500]
        ph = cp.array(ph_in, dtype='complex64')[1023::-1,200:1024+200,]
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