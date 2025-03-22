import tensorflow as tf
from typing import Tuple
from ..utils.logger import setup_logger

logger = setup_logger(name='src_data_augmentation')

def apply_augmentation(
    image: tf.Tensor, 
    label: tf.Tensor,
    rotation_prob: float = 0.5,
    flip_horizontal_prob: float = 0.5,
    flip_vertical_prob: float = 0.5,
    brightness_delta: float = 0.1,
    contrast_range: float = 0.1
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply simple data augmentation to an MRI image.
    
    Parameters:
    -----------
    image : tf.Tensor
        Input image tensor
    label : tf.Tensor
        Input label tensor
    rotation_prob : float, default=0.5
        Probability of applying random rotation
    flip_horizontal_prob : float, default=0.5
        Probability of horizontal flip
    flip_vertical_prob : float, default=0.5
        Probability of vertical flip
    brightness_delta : float, default=0.1
        Maximum brightness adjustment
    contrast_range : float, default=0.1
        Contrast adjustment range (applied as 1±range)
        
    Returns:
    --------
    Tuple[tf.Tensor, tf.Tensor]
        Augmented image and unchanged label
    """
    # Random rotation (90, 180, or 270 degrees)
    if tf.random.uniform(shape=[]) < rotation_prob:
        k = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
    
    # Random horizontal flip
    if tf.random.uniform(shape=[]) < flip_horizontal_prob:
        image = tf.image.flip_left_right(image)
    
    # Random vertical flip
    if tf.random.uniform(shape=[]) < flip_vertical_prob:
        image = tf.image.flip_up_down(image)
    
    # Brightness adjustment
    if brightness_delta > 0:
        image = tf.image.random_brightness(image, max_delta=brightness_delta)
    
    # Contrast adjustment
    if contrast_range > 0:
        image = tf.image.random_contrast(
            image, 
            lower=1.0-contrast_range, 
            upper=1.0+contrast_range
        )
    
    # Ensure pixel values stay in valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label