import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0 # type: ignore
from ..utils.logger import setup_logger
from .base_model import BaseModel

logger = setup_logger(name='src_models_transfer_models')

class TransferLearningModel(tf.keras.Model, BaseModel):
    """
    Transfer learning model for Alzheimer's disease classification.
    This class provides a framework for using pretrained models.
    """
    
    def __init__(self, 
                 base_model_name="ResNet50",
                 input_shape=(224, 224, 3),
                 num_classes=4,
                 dropout_rate=0.5,
                 freeze_base=True,
                 name="transfer_model"):
        """
        Initialize the transfer learning model.
        
        Parameters:
        -----------
        base_model_name : str
            Name of the base model ('ResNet50', 'DenseNet121', 'EfficientNetB0')
        input_shape : tuple
            Shape of input images (height, width, channels)
        num_classes : int
            Number of output classes
        dropout_rate : float
            Dropout rate for fully connected layers
        freeze_base : bool
            Whether to freeze the base model weights
        name : str
            Name of the model
        """
        super(TransferLearningModel, self).__init__(name=name)
        BaseModel.__init__(self, name=f"{base_model_name}_{name}")
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.base_model_name = base_model_name
        
        # Initialize base model
        logger.info(f"Initializing {base_model_name} as base model")
        self.base_model = self._get_base_model(base_model_name, input_shape)
        
        # Freeze/unfreeze base model
        self.base_model.trainable = not freeze_base
        if freeze_base:
            logger.info(f"Base model weights are frozen")
        else:
            logger.info(f"Base model weights are trainable")
        
        # Add custom layers on top
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(num_classes)  # No activation (logits)
    
    def _get_base_model(self, model_name, input_shape):
        """
        Get the base model architecture.
        
        Parameters:
        -----------
        model_name : str
            Name of the base model
        input_shape : tuple
            Input shape for the model
            
        Returns:
        --------
        tf.keras.Model
            Base model instance
        """
        if model_name == "ResNet50":
            base_model = ResNet50(include_top=False, 
                                weights='imagenet', 
                                input_shape=input_shape)
        elif model_name == "DenseNet121":
            base_model = DenseNet121(include_top=False, 
                                   weights='imagenet', 
                                   input_shape=input_shape)
        elif model_name == "EfficientNetB0":
            base_model = EfficientNetB0(include_top=False, 
                                       weights='imagenet', 
                                       input_shape=input_shape)
        else:
            raise ValueError(f"Unsupported base model: {model_name}")
        
        return base_model
    
    def call(self, inputs, training=None):
        """
        Forward pass for the model.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor
        training : bool
            Whether in training mode (affects dropout)
            
        Returns:
        --------
        tf.Tensor
            Output logits
        """
        # Handle grayscale (1-channel) to RGB (3-channel) conversion if needed
        if inputs.shape[-1] == 1 and self.input_shape[-1] == 3:
            x = tf.image.grayscale_to_rgb(inputs)
        else:
            x = inputs
        
        x = self.base_model(x, training=training)
        x = self.global_pool(x)
        
        if training:
            x = self.dropout1(x)
        x = self.dense1(x)
        
        if training:
            x = self.dropout2(x)
        x = self.dense2(x)
        
        x = self.output_layer(x)
        
        return x
    
    def fine_tune(self, unfreeze_layers=10):
        """
        Unfreeze the last n layers of the base model for fine-tuning.
        
        Parameters:
        -----------
        unfreeze_layers : int
            Number of layers to unfreeze from the end
        """
        # First, freeze all layers in the base model
        self.base_model.trainable = True
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Then, unfreeze the last n layers
        for layer in self.base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        logger.info(f"Fine-tuning: Unfroze the last {unfreeze_layers} layers of {self.base_model_name}")
    
    def build_graph(self):
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
    
    def save_model(self, checkpoint_dir):
        """
        Save the model to the specified checkpoint directory.
        
        Parameters:
        -----------
        checkpoint_dir : str
            Directory to save the model
            
        Returns:
        --------
        str
            Path to the saved model
        """
        model = self.build_graph()
        path = f"{checkpoint_dir}/{self.model_name}"
        model.save(path)
        logger.info(f"Model saved to {path}")
        return path


def create_transfer_model(model_name, num_classes, input_shape=(224, 224, 3), 
                        freeze_base=True, dropout_rate=0.5):
    """
    Factory function to create a transfer learning model.
    
    Parameters:
    -----------
    model_name : str
        Name of the base model ('ResNet50', 'DenseNet121', 'EfficientNetB0')
    num_classes : int
        Number of output classes
    input_shape : tuple
        Input shape for the model
    freeze_base : bool
        Whether to freeze the base model weights
    dropout_rate : float
        Dropout rate for fully connected layers
        
    Returns:
    --------
    TransferLearningModel
        Instance of the transfer learning model
    """
    logger.info(f"Creating transfer learning model with {model_name}")
    
    return TransferLearningModel(
        base_model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_base=freeze_base
    )