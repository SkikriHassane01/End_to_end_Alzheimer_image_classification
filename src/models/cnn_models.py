import tensorflow as tf
from tensorflow.keras import layers, regularizers # type: ignore
from typing import Tuple, Optional
from ..utils.logger import setup_logger
from .base_model import BaseModel
from pathlib import Path
import os
logger = setup_logger(name='src_models_cnn_models')

CHECKPOINT_DIR = Path(__file__).resolve().parents[0] / "checkpoints"

class SimpleCNN(tf.keras.Model, BaseModel):
    """
    A simple CNN model for Alzheimer's disease classification.
    
    This model consists of 4 convolutional blocks (conv + pooling)
    followed by fully connected layers for classification.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 1),
                 num_classes: int = 4,
                 name: str = "simple_cnn"):
        
        super(SimpleCNN, self).__init__(name=name)
        BaseModel.__init__(self, name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Create model layers
        self._build_model()
    
    def _build_model(self) -> None:
        """Set up the model architecture."""
        # Block 1: 32 filters
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        # Block 2: 64 filters
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        # Block 3: 128 filters
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        # Block 4: 256 filters
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool4 = layers.MaxPooling2D((2, 2))
        
        # Classification head
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(self.num_classes)  # Logits
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass for the model.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels)
        training : bool, optional
            Whether in training mode (affects dropout)
            
        Returns:
        --------
        tf.Tensor
            Output logits
        """
        # Feature extraction
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.pool4(x)
        
        # Classification
        x = self.flatten(x)
        
        if training:
            x = self.dropout1(x)
        x = self.dense1(x)
        
        if training:
            x = self.dropout2(x)
        x = self.dense2(x)
        
        return x
    
    def build_graph(self) -> tf.keras.Model:
        """
        Build the model graph for visualization.
        
        Returns:
        --------
        tf.keras.Model
            Keras model instance
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model to
        """
        model = self.build_graph()
        
        # Check if the path already has an extension
        if path.endswith('.keras') or path.endswith('.h5'):
            # If it does, use the path directly
            save_path = path
        else:
            # If it doesn't, append the model name
            save_path = os.path.join(path, self.model_name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        return save_path


class DeepCNN(tf.keras.Model, BaseModel):
    """
    A deeper CNN model for Alzheimer's disease classification with regularization.
    
    This model features:
    - Double convolutional layers in each block
    - Batch normalization for faster training and better generalization
    - L2 regularization to combat overfitting
    - Multiple dropout layers
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 1),
                 num_classes: int = 4,
                 l2_rate: float = 0.001,
                 name: str = "deep_cnn"):
        """
        Initialize the Deep CNN model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        num_classes : int
            Number of output classes
        l2_rate : float
            L2 regularization rate
        name : str
            Name of the model
        """
        super(DeepCNN, self).__init__(name=name)
        BaseModel.__init__(self, name=name)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_rate = l2_rate
        
        # Create model layers
        self._build_model()
    
    def _build_model(self) -> None:
        """Set up the model architecture."""
        # Common regularizer
        l2 = regularizers.l2(self.l2_rate)
        
        # Block 1: 32 filters
        self.conv1_1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.conv1_2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.batchnorm1 = layers.BatchNormalization()
        
        # Block 2: 64 filters
        self.conv2_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.conv2_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.batchnorm2 = layers.BatchNormalization()
        
        # Block 3: 128 filters
        self.conv3_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.conv3_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.batchnorm3 = layers.BatchNormalization()
        
        # Block 4: 256 filters
        self.conv4_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.conv4_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2)
        self.pool4 = layers.MaxPooling2D((2, 2))
        self.batchnorm4 = layers.BatchNormalization()
        
        # Classification head
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(0.5)
        self.dense1 = layers.Dense(512, activation='relu', kernel_regularizer=l2)
        self.batchnorm5 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(128, activation='relu', kernel_regularizer=l2)
        self.batchnorm6 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.3)
        self.dense3 = layers.Dense(self.num_classes)  # Logits
    
    def _conv_block(self, x: tf.Tensor, conv1: layers.Layer, conv2: layers.Layer, 
                   pool: layers.Layer, batchnorm: layers.Layer, training: Optional[bool]) -> tf.Tensor:
        """Process a single convolutional block."""
        x = conv1(x)
        x = conv2(x)
        x = pool(x)
        x = batchnorm(x, training=training)
        return x
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass for the model.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels)
        training : bool, optional
            Whether in training mode (affects dropout and batch normalization)
            
        Returns:
        --------
        tf.Tensor
            Output logits
        """
        # Feature extraction - four convolutional blocks
        x = self._conv_block(inputs, self.conv1_1, self.conv1_2, self.pool1, self.batchnorm1, training)
        x = self._conv_block(x, self.conv2_1, self.conv2_2, self.pool2, self.batchnorm2, training)
        x = self._conv_block(x, self.conv3_1, self.conv3_2, self.pool3, self.batchnorm3, training)
        x = self._conv_block(x, self.conv4_1, self.conv4_2, self.pool4, self.batchnorm4, training)
        
        # Classification
        x = self.flatten(x)
        
        if training:
            x = self.dropout1(x)
        x = self.dense1(x)
        x = self.batchnorm5(x, training=training)
        
        if training:
            x = self.dropout2(x)
        x = self.dense2(x)
        x = self.batchnorm6(x, training=training)
        
        if training:
            x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
    
    def build_graph(self) -> tf.keras.Model:
        """
        Build the model graph for visualization.
        
        Returns:
        --------
        tf.keras.Model
            Keras model instance
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the model to
        """
        model = self.build_graph()
        
        # Check if the path already has an extension
        if path.endswith('.keras') or path.endswith('.h5'):
            # If it does, use the path directly
            save_path = path
        else:
            # If it doesn't, append the model name
            save_path = os.path.join(path, self.model_name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        return save_path