import numpy as np
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import h5py


#######################################################################
# simple functions
#######################################################################

def gaus(arr_in, cen, sig, a=2**16, ph=0.):
    # Compute on the same backend as arr_in (cupy -> stays on GPU, numpy -> CPU).
    # Use 1D broadcasting instead of full meshgrid arrays, and keep everything in
    # float32 so a 20k x 20k grid costs ~1.7 GB instead of a 6.9 GB complex128 array.
    xp = cp.get_array_module(arr_in)
    # Coerce scalars to plain Python floats: under numpy 2.x (NEP 50) a float32 array
    # minus a numpy int64 scalar (as cen holds when N_zp comes from HDF5) promotes to
    # float64, doubling every temporary. Python floats are "weak" and keep it float32.
    cx, cy = float(cen[0]), float(cen[1])
    s2 = 2.0 * float(sig)**2
    x = xp.arange(arr_in.shape[0], dtype=xp.float32)[None, :]
    y = xp.arange(arr_in.shape[1], dtype=xp.float32)[:, None]
    g = (a * xp.exp(-((x - cx)**2 + (y - cy)**2) / s2)).astype(xp.float32, copy=False)
    if ph:
        g = (g * xp.exp(1j * ph)).astype(xp.complex64)
    return g


#######################################################################
# Plotting
#######################################################################

def make_hsv(arr_in):
    hsv = np.zeros((*arr_in.shape, 3))
    hsv[..., 0] = (np.angle(arr_in) + np.pi) / (2*np.pi)  # Hue from phase
    hsv[..., 1] = 1.0  # Full saturation
    hsv[..., 2] = np.abs(arr_in) / np.abs(arr_in).max()  # Value from amplitude
    return hsv


def plot_hsv(arr_in, extent_in=None):
    hsv = make_hsv(arr_in)
    if extent_in is None:
        plt.imshow(matplotlib.colors.hsv_to_rgb(hsv), origin='lower');
    else:
        plt.imshow(matplotlib.colors.hsv_to_rgb(hsv), origin='lower', extent=extent_in);


def guess_scalebar_size(npix, resolution):
    # give resolution in um
    scalebar_approx = resolution * npix / 10
    if scalebar_approx < 1:
        fac, unit = 1e3, 'nm'
    elif scalebar_approx >= 1000:
        fac, unit = 1e-3, 'mm'
    else:
        fac, unit = 1., 'um'

    scalebar_norm = scalebar_approx * fac

    exponent = np.floor(np.log10(scalebar_norm))
    base = scalebar_norm / (10 ** exponent)
    if base < 1.5: nice_base = 1
    elif base < 3.5: nice_base = 2
    elif base < 7.5: nice_base = 5
    else: nice_base = 10

    final_size = nice_base * (10 ** exponent)
    return final_size, final_size / resolution / fac, 0.01*npix, unit


#######################################################################
# write and print h5 filetrees
#######################################################################

def recursively_save_dict_to_h5(h5file, path, dictionary):
    for key, item in dictionary.items():
        key_path = f"{path}/{key}"
        if isinstance(item, dict):
            group = h5file.require_group(key_path)
            recursively_save_dict_to_h5(h5file, key_path, item)
        else:
            # Convert non-array scalars to numpy types for h5py
            if np.isscalar(item):
                item = np.array(item)
            h5file.create_dataset(key_path, data=item)


def print_h5_tree(filename, start_path='/', indent=0):
    def visit_func(name, obj):
        depth = name.count('/')
        prefix = '    ' * depth
        obj_type = "[Group]" if isinstance(obj, h5py.Group) else "[Dataset]"
        info = ""

        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) < 1:
                info = f"value={obj[()]}, dtype={obj.dtype}"
            else:
                info = f"shape={obj.shape}, dtype={obj.dtype}"

        print(f"{prefix}- {name.split('/')[-1]} {obj_type} {info}")

        if obj.attrs:
            for attr_key, attr_val in obj.attrs.items():
                print(f"{prefix}    @ {attr_key} = {attr_val}")

    with h5py.File(filename, 'r') as f:
        print(f"File: {filename}")
        f[start_path].visititems(visit_func)
