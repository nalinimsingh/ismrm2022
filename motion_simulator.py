from scipy import ndimage
import csv
import numpy as np
import os

from interlacer import utils
import mrimotion as mot

def apply_func_component(func,arr):
    mag_func = func(np.abs(arr))
    phase_func = func(np.angle(arr))
    real_func = mag_func*np.cos(phase_func)
    imag_func = mag_func*np.sin(phase_func)
    return real_func+1j*imag_func

def add_rotation_and_translations(sl, coord_list, angle, num_pix, sl_shape):
    """Add k-space rotations and translations to input slice.
    At each line in coord_list in k-space, induce a rotation and translation.
    Args:
      sl(float): Numpy array of shape (n, n) containing input image data
      coord_list(int): Numpy array of (num_points) k-space line indices at which to induce motion
      angle(float): Numpy array of angles by which to rotate the input image; of shape (num_points)
      num_pix(float): List of horizontal and vertical translations by which to shift the input image; of shape (num_points, 2)
    Returns:
      sl_k_corrupt(float): Motion-corrupted k-space version of the input slice, of shape(n, n)
    """
    orig_sl_shape = sl.shape
    sl = np.reshape(sl, sl_shape)
    sl_shift = np.fft.fftshift(sl)
    n = sl.shape[0]
    coord_list = np.concatenate([coord_list, [-1]])
    sl_k_true = np.fft.fftshift(np.fft.fft2(sl_shift))

    sl_k_combined = np.zeros(sl.shape, dtype='complex64')
    sl_k_combined[:,:coord_list[0]] = sl_k_true[:,:coord_list[0]]  

    for i in range(len(coord_list) - 1):
        rotate_func = lambda x: ndimage.rotate(x, angle[i], reshape=False, mode='nearest')
        sl_rotate = apply_func_component(rotate_func, sl)

        if(len(num_pix.shape) == 1):
            sl_moved = ndimage.interpolation.shift(
                sl_rotate, [0, num_pix[i]], mode='nearest')
        elif(num_pix.shape[1] == 2):
            sl_moved = ndimage.interpolation.shift(
                sl_rotate, [0, num_pix[i, 0]])
            sl_moved = ndimage.interpolation.shift(
                sl_moved, [num_pix[i, 1], 0])

        sl_moved_shift = np.fft.fftshift(sl_moved)
        sl_k_after = np.fft.fftshift(np.fft.fft2(sl_moved_shift))

        if(coord_list[i + 1] != -1):
            sl_k_combined[:,coord_list[i]:coord_list[i + 1]
                          ] = sl_k_after[:,coord_list[i]:coord_list[i + 1]]
            if(coord_list[i] <= int(n / 2) and int(n / 2) < coord_list[i + 1]):
                sl_k_true = sl_k_after
        else:
            sl_k_combined[:,coord_list[i]:] = sl_k_after[:,coord_list[i]:]
            if(coord_list[i] <= int(n / 2)):
                sl_k_true = sl_k_after

    sl_k_combined = np.reshape(sl_k_combined, orig_sl_shape)
    sl_k_true = np.reshape(sl_k_true, orig_sl_shape)
    return sl_k_combined, sl_k_true 

def sim_motion(kspace, sl, inter, num_points=10, max_htrans=0.03, max_vtrans=0.03, max_rot=0.03):

    n_dim = len(kspace.shape)
    if(n_dim==4):
        kspace = np.expand_dims(kspace,-1)
    if(n_dim==6):
        kspace = kspace[...,0]

    n_y, n_x, n_sl, n_coil, n_inter = kspace.shape

    # Select a slice
    # Randomly pick a slice since this is all 2D.
    # And avoid slices on the ends

    k_slice = np.fft.ifftshift(kspace[:,:,sl,:,inter],axes=(0,1))
    img_slice = np.fft.fftshift(np.fft.ifftn(k_slice,axes=(0,1)),axes=(0,1))

    coord_list = np.sort(
        np.random.choice(
            n_x,
            size=num_points,
            replace=False))
    num_pix = np.zeros((num_points, 2))
    angle = np.zeros(num_points)

    max_htrans_pix = n_x * max_htrans
    max_vtrans_pix = n_y * max_vtrans
    max_rot_deg = 360 * max_rot

    num_pix[:, 0] = np.random.random(
        num_points) * (2 * max_htrans_pix) - max_htrans_pix
    num_pix[:, 1] = np.random.random(
        num_points) * (2 * max_vtrans_pix) - max_vtrans_pix
    angle = np.random.random(num_points) * \
        (2 * max_rot_deg) - max_rot_deg

    img_sl_flat = np.reshape(img_slice, (-1, img_slice.shape[-1]))
    sl_shape = img_slice.shape[:-1]
    coil_motion_func = lambda sl: add_rotation_and_translations(sl, coord_list, angle, num_pix, sl_shape)
    k_corrupt, k_true  = np.apply_along_axis(coil_motion_func, 0, img_sl_flat)
    k_corrupt = np.reshape(k_corrupt, img_slice.shape)
    k_true = np.reshape(k_true, img_slice.shape)
    return k_corrupt, k_true

def generate_motion_corrupted_brain_data(scan_list_path):
    acq_paths = []
    with open(scan_list_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            acq_paths.append(row[0])

    for acq_path in acq_paths:
        acq_data_path = os.path.join(acq_path,'kspace_acquired.npz')
        kspace = mot.utils.load_kspace(acq_data_path, 'ge', z_shift=False)

        n_sl = kspace.shape[2]
        for sl in range(3,n_sl-3):
            for inter in range(2):
                k_corrupt, k_true = sim_motion(kspace,sl,inter)

                # Split real and imaginary
                k_corrupt = utils.split_reim(k_corrupt)
                k_true = utils.split_reim(k_true)

                # Keep only the first 44 channels
                k_corrupt = k_corrupt[...,:44,:]
                k_true = k_true[...,:44,:]

                # Combine all channels
                new_shape = k_corrupt.shape[:2]+(-1,)
                k_corrupt = np.reshape(k_corrupt, new_shape)
                k_true = np.reshape(k_true, new_shape)

                # Add batch dimension
                k_corrupt = np.expand_dims(k_corrupt, 0)
                k_true = np.expand_dims(k_true, 0)

                yield(k_corrupt, k_true)
