import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.utils import get_custom_objects

from interlacer import layers, utils
import importlib
importlib.reload(layers)


def reconstruct_rss(img_arr):
    comp_img_arr = utils.join_reim_channels(img_arr)
    sq = K.square(K.abs(comp_img_arr))
    return K.sqrt(K.sum(sq,axis=3))

def get_multicoil_interlacer_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_convs,
        num_layers,
        num_coils):
    """Interlacer model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size, n, n, 2)) and returns
    a frequency-space output of the same size, comprised of interlacer layers and with connections
    from the input to each layer. Handles variable input size, and crops to a 320x320 image at the end.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_convs(int): Number of convolutions per layer
      num_layers(int): Number of convolutional layers in model
      enforce_dc(Bool): Whether to paste in original acquired k-space lines in final output

    Returns:
      model: Keras model comprised of num_layers core interlaced layers with specified nonlinearities

    """
    inputs = Input(input_size)

    inp_conv = layers.BatchNormConv(
            num_features, 1)(inputs)

    
    inp_img = utils.convert_channels_to_image_domain(inputs)
    inp_img_unshift = tf.signal.ifftshift(inp_img, axes=(1,2))
    inp_img_conv_unshift = layers.BatchNormConv(
            num_features, 1)(inp_img_unshift)
    inp_img_conv = tf.signal.ifftshift(inp_img_conv_unshift, axes=(1,2))
    '''
    inp_img = utils.convert_channels_to_image_domain(inputs)
    inp_img_conv = layers.BatchNormConv(
            num_features, 1)(inp_img)
    '''
 
    freq_in = inputs
    img_in = inp_img

    for i in range(num_layers):
        img_conv, k_conv = layers.Interlacer(
            num_features, kernel_size, num_convs, shift=False)([img_in, freq_in])

        freq_in = k_conv + inp_conv
        img_in = img_conv + inp_img_conv

    output = Conv2D(2*num_coils, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(freq_in) + inputs

    model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def get_multicoil_conv_model(
        input_size,
        nonlinearity,
        kernel_size,
        num_features,
        num_layers,
        num_coils):
    """Alternating model with residual convolutions.

    Returns a model that takes a frequency-space input (of shape (batch_size,
    n, n, num_coils*2)) and returns a frequency-space output of the same size, comprised of alternating frequency- and image-space convolutional layers and with connections from the input to each layer.

    Args:
      input_size(int): Tuple containing input shape, excluding batch size
      nonlinearity(str): 'relu' or '3-piece'
      kernel_size(int): Dimension of each convolutional filter
      num_features(int): Number of features in each intermediate network layer
      num_layers(int): Number of convolutional layers in model
      num_coils(int): Number of coils in final output

    Returns:
      model: Keras model comprised of num_layers alternating image- and frequency-space convolutional layers with specified nonlinearities

    """
    inputs = Input(input_size)
    
    inp_conv = layers.BatchNormConv(
            num_features, 1)(inputs)

    prev_layer = inputs

    for i in range(num_layers):
        conv = layers.BatchNormConv(
            num_features, kernel_size)(prev_layer)
        nonlinear = layers.get_nonlinear_layer(nonlinearity)(conv)
        prev_layer = nonlinear+inp_conv

    output = Conv2D(num_coils*2, kernel_size, activation=None, padding='same',
                    kernel_initializer='he_normal')(prev_layer) + inputs

    model = keras.models.Model(inputs=inputs, outputs=output)
    return model
