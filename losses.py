import sys

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K

from interlacer import utils
try:
    import lpips_tf
except:
    pass


def reconstruct_rss(img_arr):
    comp_img_arr = utils.join_reim_channels(img_arr)
    sq = K.square(K.abs(comp_img_arr))
    return K.sqrt(K.sum(sq,axis=3))

def prep_for_multicoil_loss(tensor):
    tensor = utils.convert_channels_to_image_domain(tensor)
    tensor = tf.signal.fftshift(tensor, axes=(1,2))
    tensor = reconstruct_rss(tensor)
    tensor = tf.expand_dims(tensor, -1)
    
    return tensor

def prep_img_for_multicoil_loss(tensor):
    tensor = tf.signal.fftshift(tensor,axes=(1,2))
    tensor = reconstruct_rss(tensor)
    tensor = tf.expand_dims(tensor, -1)
    
    return tensor


def multicoil_ssim(output_domain, num_coils):
    """Specifies a function which computes the appropriate loss function.

    Loss function here is SSIM on image-space data.

    Args:
      output_domain(str): Network output domain ('FREQ' or 'IMAGE')

    Returns:
      Function computing loss value from a true and predicted input

    """
    if(output_domain == 'IMAGE_COILCOMBINED'):
        def image_ssim_multicoil(y_true, y_pred):
            y_true.set_shape((None,None,None,2))
            
            y_true = prep_imgcc_for_multicoil_loss(y_true)
            y_pred = prep_imgcc_for_multicoil_loss(y_pred)

            return -1 * tf.image.ssim(y_true, y_pred,
                                      max_val=K.max(y_true), filter_size=7)
        return image_ssim_multicoil
    elif(output_domain == 'IMAGE'):
        def image_ssim_multicoil(y_true, y_pred):
            y_true.set_shape((None,None,None,num_coils*2))
            
            y_true = prep_img_for_multicoil_loss(y_true)
            y_pred = prep_img_for_multicoil_loss(y_pred)

            return -1 * tf.image.ssim(y_true, y_pred,
                                      max_val=K.max(y_true), filter_size=7)
        return image_ssim_multicoil        
    elif(output_domain == 'FREQ'):
        def image_ssim_multicoil(y_true, y_pred):
            y_true.set_shape((None,None,None,num_coils*2))
            
            y_true = prep_for_multicoil_loss(y_true)
            y_pred = prep_for_multicoil_loss(y_pred)

            return -1 * tf.image.ssim(y_true, y_pred,
                                      max_val=K.max(y_true), filter_size=7)
        return image_ssim_multicoil