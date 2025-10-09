# ============================================================================
# MODEL_ARCHITECTURE.PY - Enhanced U-Net with Attention Gates
# ============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import sys

# Increase recursion limit for deep model
sys.setrecursionlimit(50000)

# Import config (handle import error gracefully)
try:
    import config.config as cfg
    IMG_HEIGHT = cfg.IMG_HEIGHT
    IMG_WIDTH = cfg.IMG_WIDTH
    IMG_CHANNELS = cfg.IMG_CHANNELS
except:
    # Default values if config import fails
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3


def attention_block(x, g, inter_channel):
    """Attention Gate for U-Net"""
    theta_x = layers.Conv2D(inter_channel, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_channel, 1, strides=1, padding='same')(g)
    
    if x.shape[1] != g.shape[1]:
        phi_g = layers.UpSampling2D(size=(2, 2))(phi_g)
    
    add_xg = layers.Add()([theta_x, phi_g])
    act_xg = layers.Activation('relu')(add_xg)
    
    psi = layers.Conv2D(1, 1, padding='same')(act_xg)
    psi = layers.Activation('sigmoid')(psi)
    
    y = layers.Multiply()([x, psi])
    y = layers.Conv2D(inter_channel, 1, padding='same')(y)
    
    return y


def residual_conv_block(inputs, num_filters, use_dropout=False):
    """Residual Convolutional Block"""
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if use_dropout:
        x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if inputs.shape[-1] == num_filters:
        shortcut = inputs
    else:
        shortcut = layers.Conv2D(num_filters, 1, padding='same')(inputs)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x


def encoder_block(inputs, num_filters, use_dropout=False):
    """Encoder block with residual connections"""
    x = residual_conv_block(inputs, num_filters, use_dropout)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters, use_attention=True):
    """Decoder block with attention gates"""
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    
    if use_attention:
        skip_features = attention_block(skip_features, x, num_filters)
    
    x = layers.Concatenate()([x, skip_features])
    x = residual_conv_block(x, num_filters)
    return x


def build_enhanced_unet(input_shape=None):
    """
    Build Enhanced U-Net Architecture
    THIS IS THE MAIN FUNCTION
    """
    if input_shape is None:
        input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    inputs = layers.Input(input_shape, name='input_image')
    
    # ENCODER
    s1, p1 = encoder_block(inputs, 64, use_dropout=False)
    s2, p2 = encoder_block(p1, 128, use_dropout=True)
    s3, p3 = encoder_block(p2, 256, use_dropout=True)
    s4, p4 = encoder_block(p3, 512, use_dropout=True)
    
    # BRIDGE
    bridge = residual_conv_block(p4, 1024, use_dropout=True)
    
    # DECODER
    d1 = decoder_block(bridge, s4, 512, use_attention=True)
    d2 = decoder_block(d1, s3, 256, use_attention=True)
    d3 = decoder_block(d2, s2, 128, use_attention=True)
    d4 = decoder_block(d3, s1, 64, use_attention=True)
    
    # OUTPUT
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid', 
                           dtype='float32', name='output_mask')(d4)
    
    model = models.Model(inputs, outputs, name='Enhanced-Attention-UNet')
    
    return model


# Test if this file works when run directly
if __name__ == "__main__":
    print("Testing model architecture...")
    model = build_enhanced_unet()
    print(f"✓ Model created successfully!")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")