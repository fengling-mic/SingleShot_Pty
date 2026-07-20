import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation


def hann2d(shape):
    wy = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    return np.outer(wy, wx)


def register_translation(img_reference_in, img_to_align_in,
                         upsample_factor=100,
                         use_phase_only=False,
                         use_window=False):
    """
    Register two images to subpixel accuracy using phase correlation.

    Parameters:
        img_reference (ndarray): Reference image (2D)
        img_to_align (ndarray): Image to align (2D)
        upsample_factor (int): Controls subpixel accuracy. Higher = finer.

    Returns:
        aligned_img (ndarray): Aligned version of img_to_align
        shift_values (tuple): (shift_y, shift_x) subpixel shift
    """

    if use_window:
        w = hann2d(img_reference_in.shape)
        img_reference = img_reference_in * w
        img_to_align = img_to_align_in * w
    else:
        img_reference = np.copy(img_reference_in)
        img_to_align = np.copy(img_to_align_in)

    if use_phase_only:
        ref = np.angle(img_reference)
        mov = np.angle(img_to_align)
    else:
        ref = img_reference
        mov = img_to_align

    shift_values, error, _ = phase_cross_correlation(
        ref, mov, upsample_factor=upsample_factor)

    aligned_img = shift(img_to_align_in, shift=shift_values, mode='grid-wrap')

    return aligned_img, np.array(shift_values)
