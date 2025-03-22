import tensorflow as tf
from ..utils.logger import setup_logger

logger = setup_logger(name='src_data_preprocessing')

def preprocess_image(image: tf.Tensor, 
                    target_height: int = 224, 
                    target_width: int = 224,
                    include_brain_extraction: bool = False) -> tf.Tensor:
    """
    Apply comprehensive preprocessing to an MRI image.
    
    Parameters:
    -----------
    image : tf.Tensor
        Input image tensor
a    target_height : int, default=224
        Target height for resizing
    target_width : int, default=224
        Target width for resizing
    include_brain_extraction : bool, default=False
        Whether to apply brain extraction (skull stripping)
        
    Returns:
    --------
    tf.Tensor
        Preprocessed image tensor
    """
    # Ensure image is float type
    image = tf.cast(image, tf.float32)
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Apply brain extraction if requested
    if include_brain_extraction:
        # Simple threshold-based approach for skull stripping
        threshold = tf.reduce_mean(image) * 0.8
        mask = tf.cast(image > threshold, tf.float32)
        image = image * mask
    
    # Resize with padding to maintain aspect ratio
    image = tf.image.resize_with_pad(
        image, 
        target_height, 
        target_width,
        method=tf.image.ResizeMethod.BILINEAR
    )
    
    # Z-score normalization (standardization)
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image = (image - mean) / (std + 1e-7)  # Adding small epsilon to avoid division by zero
    
    return image