import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, List
from ..utils.logger import setup_logger

logger = setup_logger(name='src_training_class_balancing')

def balance_dataset(dataset: tf.data.Dataset, 
                   class_names: List[str],
                   target_samples_per_class: int = 10000,
                   batch_size: int = 32) -> Tuple[tf.data.Dataset, Dict[int, float]]:
    """
    Balance an imbalanced dataset through strategic downsampling and class weighting.
    
    Parameters:
    -----------
    dataset : tf.data.Dataset
        The original imbalanced dataset
    class_names : List[str]
        List of class names in order of their indices
    target_samples_per_class : int
        Target number of samples for each class (default: 10000)
        Classes with fewer samples than this will be kept as is
    batch_size : int
        Batch size for the returned dataset
        
    Returns:
    --------
    Tuple[tf.data.Dataset, Dict[int, float]]
        Balanced dataset and class weights dictionary for loss function
    """
    logger.info("Balancing dataset through strategic downsampling and class weighting...")
    
    # TODO: Step 1: Compute class distribution
    class_counts = {i: 0 for i in range(len(class_names))}
    
    # Unbatch the dataset if it's batched
    if hasattr(dataset, '_batch_size') and dataset._batch_size is not None:
        dataset = dataset.unbatch()
    
    # Count samples per class
    for _, labels in dataset:
        class_idx = tf.argmax(labels).numpy()
        class_counts[int(class_idx)] += 1
    
    logger.info(f"==>Original class distribution: {class_counts}")
    
    # TODO: Step 2: Calculate class weights
    # the class weights help to balance the loss function
    
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    
    class_weights = {}
    for class_idx, count in class_counts.items():
        class_weights[class_idx] = total_samples / (n_classes * count)

    logger.info(f"==>Class weights: {class_weights}")
    
    # TODO: Step 3: Balance through downsampling
    class_datasets = []
    
    for class_idx in range(len(class_names)):
        # Filter dataset to include only current class
        class_ds = dataset.filter(
            lambda x, y: tf.equal(tf.argmax(y), class_idx)
        )
        
        # Determine sample count
        sample_count = class_counts[class_idx]
        
        # For majority classes, downsample
        if sample_count > target_samples_per_class:
            # Shuffle before taking subset to ensure randomness
            class_ds = class_ds.shuffle(buffer_size=sample_count, seed=42)
            class_ds = class_ds.take(target_samples_per_class)
            logger.info(f"==> Downsampled class {class_names[class_idx]} from {sample_count} to {target_samples_per_class}")
        else:
            logger.info(f"==> Kept all {sample_count} samples for class {class_names[class_idx]}")
        
        class_datasets.append(class_ds)
    
    # Step 4: Combine all datasets
    balanced_ds = class_datasets[0]
    for ds in class_datasets[1:]:
        balanced_ds = balanced_ds.concatenate(ds)
    
    # Step 5: Shuffle, batch, and prefetch for performance
    balanced_ds = balanced_ds.shuffle(buffer_size=10000, seed=42)
    balanced_ds = balanced_ds.batch(batch_size)
    balanced_ds = balanced_ds.prefetch(tf.data.AUTOTUNE)
    
    # Count samples in balanced dataset
    balanced_count = 0
    for _ in balanced_ds:
        balanced_count += 1
    balanced_count *= batch_size  # Approximate due to potential incomplete last batch
    
    logger.info(f"Total samples in balanced dataset: ~{balanced_count}")
    logger.info("Dataset balancing complete")
    
    return balanced_ds, class_weights

def create_weighted_loss(class_weights: Dict[int, float]) -> callable:
    """
    Create a weighted categorical crossentropy loss function.
    
    Parameters:
    -----------
    class_weights : Dict[int, float]
        Dictionary mapping class indices to their weights
        
    Returns:
    --------
    callable
        Weighted loss function
    """
    def weighted_categorical_crossentropy(y_true, y_pred):
        # Convert class weights to a tensor
        weights_tensor = tf.constant([class_weights[i] for i in range(len(class_weights))])
        
        # Apply class weights to the loss
        unweighted_losses = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        
        # Get the class index for each sample
        weights = tf.reduce_sum(y_true * weights_tensor, axis=1)
        
        # Apply weights to the loss
        weighted_losses = unweighted_losses * weights
        
        return tf.reduce_mean(weighted_losses)
    
    return weighted_categorical_crossentropy