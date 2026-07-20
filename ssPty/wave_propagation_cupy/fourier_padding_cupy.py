import numpy as np
import cupy as cp
from ssPty.misc.fft_wrapper_cupy import fft2_with_shift, ifft2_with_shift, fft2_with_shift_3d, ifft2_with_shift_3d


#######################################################################
# Fourier padding and cropping
#######################################################################
def fourier_pad(arr_in, shape_out, fft_plan_unpad, fft_plan_pad):
    '''
    Fourier padding for RPI algorithm
    '''
    arr_fft = fft2_with_shift(arr_in, fft_plan_unpad)

    # Create a padded version of obj_fft
    arr_pad_fft = cp.zeros((shape_out[0], shape_out[1]), dtype='complex64')
    center = (shape_out[0] // 2, shape_out[1] // 2)
    arr_center = (arr_fft.shape[0] // 2, arr_fft.shape[1] // 2)
    
    # Place obj_fft in the center of obj_pad_fft
    arr_pad_fft[center[0] - arr_center[0]:center[0] + arr_center[0],
                center[1] - arr_center[1]:center[1] + arr_center[1]] = arr_fft
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift(arr_pad_fft, fft_plan_pad)


def fourier_crop(arr_in, shape_out, fft_plan_pad, fft_plan_unpad):
    '''
    Fourier cropping for RPI algorithm
    '''
    arr_fft = fft2_with_shift(arr_in, fft_plan_pad)

    # Create a cropped version of arr_fft
    center_in = (arr_fft.shape[0] // 2, arr_fft.shape[1] // 2)
    center_out = (shape_out[0] // 2, shape_out[1] // 2)
    
    # Extract the central part of arr_fft
    arr_crop_fft = arr_fft[center_in[0] - center_out[0]:center_in[0] + center_out[0],
                           center_in[1] - center_out[1]:center_in[1] + center_out[1]]
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift(arr_crop_fft, fft_plan_unpad)


def fourier_pad_3d(arr_in, shape_out, fft_plan_unpad, fft_plan_pad):
    '''
    Fourier padding for RPI algorithm
    '''
    axes_in = (1, 2)
    arr_fft = fft2_with_shift_3d(arr_in, fft_plan_unpad, axes_in)

    # Create a padded version of obj_fft
    arr_pad_fft = cp.zeros((arr_in.shape[0], shape_out[0], shape_out[1]), dtype='complex64')
    center = (shape_out[0] // 2, shape_out[1] // 2)
    arr_center = (arr_fft.shape[1] // 2, arr_fft.shape[2] // 2)
    
    # Place obj_fft in the center of obj_pad_fft
    arr_pad_fft[:,center[0] - arr_center[0]:center[0] + arr_center[0],
                  center[1] - arr_center[1]:center[1] + arr_center[1]] = arr_fft
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift_3d(arr_pad_fft, fft_plan_pad, axes_in)


def fourier_pad_4d(arr_in, shape_out, fft_plan_unpad, fft_plan_pad):
    '''
    Fourier padding for RPI algorithm
    '''
    axes_in = (2, 3)
    arr_fft = fft2_with_shift_3d(arr_in, fft_plan_unpad, axes_in)

    # Create a padded version of obj_fft
    arr_pad_fft = cp.zeros((arr_in.shape[0], arr_in.shape[1], shape_out[0], shape_out[1]), dtype='complex64')
    center = (shape_out[0] // 2, shape_out[1] // 2)
    arr_center = (arr_fft.shape[2] // 2, arr_fft.shape[3] // 2)
    
    # Place obj_fft in the center of obj_pad_fft
    arr_pad_fft[:,:,center[0] - arr_center[0]:center[0] + arr_center[0],
                    center[1] - arr_center[1]:center[1] + arr_center[1]] = arr_fft
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift_3d(arr_pad_fft, fft_plan_pad, axes_in)


def fourier_crop_3d(arr_in, shape_out, fft_plan_pad, fft_plan_unpad):
    '''
    Fourier cropping for RPI algorithm
    '''
    axes_in = (1, 2)
    arr_fft = fft2_with_shift_3d(arr_in, fft_plan_pad, axes_in)

    # Create a cropped version of arr_fft
    center_in = (arr_fft.shape[1] // 2, arr_fft.shape[2] // 2)
    center_out = (shape_out[0] // 2, shape_out[1] // 2)
    
    # Extract the central part of arr_fft
    arr_crop_fft = arr_fft[:,center_in[0] - center_out[0]:center_in[0] + center_out[0],
                             center_in[1] - center_out[1]:center_in[1] + center_out[1]]
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift_3d(cp.ascontiguousarray(arr_crop_fft), fft_plan_unpad, axes_in)


def fourier_crop_4d(arr_in, shape_out, fft_plan_pad, fft_plan_unpad):
    '''
    Fourier cropping for RPI algorithm
    '''
    axes_in = (2, 3)
    arr_fft = fft2_with_shift_3d(arr_in, fft_plan_pad, axes_in)

    # Create a cropped version of arr_fft
    center_in = (arr_fft.shape[2] // 2, arr_fft.shape[3] // 2)
    center_out = (shape_out[0] // 2, shape_out[1] // 2)
    
    # Extract the central part of arr_fft
    arr_crop_fft = arr_fft[:,:,center_in[0] - center_out[0]:center_in[0] + center_out[0],
                               center_in[1] - center_out[1]:center_in[1] + center_out[1]]
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift_3d(cp.ascontiguousarray(arr_crop_fft), fft_plan_unpad, axes_in)


def fourier_pad_nd(arr_in, shape_out, fft_plan_unpad, fft_plan_pad):
    '''
    Fourier padding for RPI algorithm
    '''
    axes_in = (-2, -1)
    arr_fft = fft2_with_shift_3d(arr_in, fft_plan_unpad, axes_in)

    # Create a padded version of obj_fft
    arr_pad_fft = cp.zeros(list(arr_in.shape[:-2]) + list(shape_out), dtype='complex64')
    center = (shape_out[0] // 2, shape_out[1] // 2)
    arr_center = (arr_fft.shape[-2] // 2, arr_fft.shape[-1] // 2)
    
    # Place obj_fft in the center of obj_pad_fft
    arr_pad_fft[...,
                center[0] - arr_center[0]:center[0] + arr_center[0],
                center[1] - arr_center[1]:center[1] + arr_center[1]] = arr_fft
    
    # Inverse Fourier transform back to spatial domain
    return ifft2_with_shift_3d(arr_pad_fft, fft_plan_pad, axes_in)