import tensorflow as tf  # type: ignore
from abc import ABC, abstractmethod
import os 

class BaseModel(ABC):
    """
    Abstract base class for all models.
    Provides a common functionality and enforces implementation or required methods
    """
    
    def __init__(self, name="base_model"):
        """
        Initializes the base model
        
        Arg:
            name: str, name of the model
        """
        super(BaseModel, self).__init__()
        self.model_name = name
        
    @abstractmethod
    def call(self, inputs, training=None):
        """
        Forward pass of the model 
        Must be implemented by the subclass
        
        Args:
            inputs: tf.Tensor, input tensor
            training: bool, whether the model is in training mode or not
        
        Returns:
            output: tf.Tensor, output tensor
        """
        pass

    @abstractmethod
    def save_model(self, checkpoint_dir):
        """
        Save the model to the specified checkpoint directory.
        
        Parameters:
        ----------
        checkpoint_dir : str
            Directory to save the model
            
        Returns:
        -------
        output : str
            Path to the saved model
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, self.model_name)
        self.save(path)
        return path

    def load_model(self, checkpoint_path):
        """
        Load the model from the specified checkpoint path.
        
        Parameters:
        ----------
        checkpoint_path : str
            Path to the model checkpoint
            
        Returns:
        -------
        BaseModel
            Loaded model
        """
        self.load_weights(checkpoint_path)
        return self
    
    def predict_with_softmax(self, inputs):
        """
        Make prediction with softmax activation.
        
        Parameters:
        ----------
        inputs : tf.Tensor
            Input tensor
            
        Returns:
        -------
        tf.Tensor
            Probability distribution over classes
        """
        predictions = self(inputs, training=False)
        return tf.nn.softmax(predictions)