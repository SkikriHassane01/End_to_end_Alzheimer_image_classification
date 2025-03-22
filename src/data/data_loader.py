import os
import tensorflow as tf
from pathlib import Path
from typing import Dict
from ..utils.logger import setup_logger

logger = setup_logger(name='src_data_data_loader')

DATA_DIR = Path(__file__).resolve().parents[2] / 'Data' / 'raw'

class AlzheimerDataset:
    """
    Dataset class for loading and managing Alzheimer's MRI images.
    """
    
    def __init__(self, 
                 data_dir: str = DATA_DIR,
                 img_height: int = 224,
                 img_width: int = 224,
                 batch_size: int = 32, # process 32 images at a time, not all at once
                 validation_split: float = 0.2,
                 test_split: float = 0.1,
                 seed: int = 42):
        """
        Initialize the Alzheimer's dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing class subdirectories of images
        img_height : int
            Target image height
        img_width : int
            Target image width
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        test_split : float
            Fraction of data to use for testing
        seed : int
            Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.seed = seed
        
        # Verify data directory exists
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get class names from directory structure
        self.class_names = [item.name for item in os.scandir(self.data_dir) if item.is_dir()]
        self.num_classes = len(self.class_names)
        
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.class_names)}")
        
        # Class index mapping
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}
        
    def create_dataset(self, augment: bool = True) -> Dict[str, tf.data.Dataset]:
        """
        Create TensorFlow datasets for training, validation, and testing.
        
        Parameters:
        -----------
        augment : bool
            Whether to apply data augmentation to the training set
            
        Returns:
        --------
        Dict[str, tf.data.Dataset]
            Dictionary containing 'train', 'validation', and 'test' datasets
        """
        # Load all images with labels
        logger.info("Creating dataset...")
        
        # Create the base dataset from the directory
        all_ds = tf.keras.utils.image_dataset_from_directory( # return (images, labels) tuples
            self.data_dir,
            shuffle=True, # mix up the data so the model doesn't learn the order
            image_size=(self.img_height, self.img_width), # resize the images to 224x224
            batch_size=None,  # We'll batch later after splitting
            seed=self.seed,
            color_mode='grayscale', # single channel images
            label_mode='categorical' # One-hot encoded labels [0,1,0,0] for 4 classes
        )
        
        # Get dataset size
        dataset_size = tf.data.experimental.cardinality(all_ds).numpy() # get the number of images
        logger.info(f"Total dataset size: {dataset_size} images")
        
        # Calculate split sizes
        val_size = int(self.validation_split * dataset_size)
        test_size = int(self.test_split * dataset_size)
        train_size = dataset_size - val_size - test_size
        
        logger.info(f"Split sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
        
        # Split the dataset
        train_ds = all_ds.take(train_size)
        remaining_ds = all_ds.skip(train_size)
        val_ds = remaining_ds.take(val_size)
        test_ds = remaining_ds.skip(val_size)
        
        # Apply preprocessing and augmentation
        if augment:
            logger.info("Applying data augmentation to training set")
            from .augmentation import apply_augmentation
            train_ds = train_ds.map(apply_augmentation, 
                                  num_parallel_calls=tf.data.AUTOTUNE)
        
        # Apply preprocessing to all datasets
        from .preprocessing import preprocess_image
        train_ds = train_ds.map(lambda x, y: (preprocess_image(x), y), 
                              num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (preprocess_image(x), y), 
                            num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.map(lambda x, y: (preprocess_image(x), y), 
                             num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch for performance
        train_ds = train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        datasets = {
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        }
        
        logger.info("Dataset creation complete")
        return datasets
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced classes.
        
        Returns:
        --------
        Dict[int, float]
            Dictionary mapping class indices to their weights
        """
        # Count samples per class
        class_counts = {}
        for cls in self.class_names:
            class_dir = os.path.join(self.data_dir, cls)
            num_samples = len([f for f in os.listdir(class_dir) 
                              if os.path.isfile(os.path.join(class_dir, f))])
            class_counts[self.class_indices[cls]] = num_samples
        
        # Calculate weights
        total_samples = sum(class_counts.values())
        class_weights = {
            cls_idx: total_samples / (len(class_counts) * count) 
            for cls_idx, count in class_counts.items()
        }
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights