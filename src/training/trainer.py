import tensorflow as tf
import os
import time
from typing import Dict, List
from pathlib import Path
from ..utils.logger import setup_logger
from ..models.base_model import BaseModel
from .class_balancing import create_weighted_loss, balance_dataset

logger = setup_logger(name='trainer')

class Trainer:
    """
    Simple trainer class for Alzheimer's disease classification models.
    """
    
    def __init__(self, 
                 model: BaseModel,
                 train_dataset: tf.data.Dataset,
                 val_dataset: tf.data.Dataset,
                 class_names: List[str],
                 learning_rate: float = 0.001,
                 checkpoint_dir: str = './checkpoints',
                 tensorboard_dir: str = './logs'):
        """
        Initialize the trainer with a model and datasets.
        
        Parameters:
        -----------
        model : BaseModel
            Model to train
        train_dataset : tf.data.Dataset
            Training dataset
        val_dataset : tf.data.Dataset
            Validation dataset
        class_names : List[str]
            Names of the classes
        learning_rate : float
            Initial learning rate
        checkpoint_dir : str
            Directory to save model checkpoints
        tensorboard_dir : str
            Directory to save TensorBoard logs
        """
        # Step 1: Store basic parameters
        self.model = model
        self.raw_train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.class_names = class_names
        self.learning_rate = learning_rate
        
        # Step 2: Set up directories
        self.checkpoint_dir = Path(checkpoint_dir) / model.model_name
        self.tensorboard_dir = Path(tensorboard_dir) / model.model_name
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        # Step 3: Set up optimizer and metrics
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.val_accuracy = tf.keras.metrics.CategoricalAccuracy()
        self.train_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()
        
        # Step 4: Initialize balanced dataset and loss function
        self.train_dataset = None
        self.class_weights = None
        self.loss_fn = None
        
        # Step 5: Set up TensorBoard
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.tensorboard_writer = tf.summary.create_file_writer(
            str(self.tensorboard_dir / current_time)
        )
        
        logger.info(f"Trainer initialized for model: {model.model_name}")
    
    def prepare_training_data(self, target_samples_per_class: int = 10000, batch_size: int = 32):
        """
        Prepare training data by balancing classes and setting up the loss function.
        
        Parameters:
        -----------
        target_samples_per_class : int
            Target number of samples per class for downsampling
        batch_size : int
            Batch size for the balanced dataset
        """
        logger.info("Preparing training data...")
        
        # Step 1: Balance the dataset and get class weights
        self.train_dataset, self.class_weights = balance_dataset(
            dataset=self.raw_train_dataset,
            class_names=self.class_names,
            target_samples_per_class=target_samples_per_class,
            batch_size=batch_size
        )
        
        # Step 2: Create weighted loss function using class weights
        self.loss_fn = create_weighted_loss(self.class_weights)
        
        logger.info("Training data preparation complete")
    
    def train_step(self, images, labels):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(images, training=True)
            
            # Calculate loss
            loss = self.loss_fn(labels, predictions)
        
        # Backpropagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)
        
        return loss.numpy()
    
    def val_step(self, images, labels):
        """Perform a single validation step."""
        # Forward pass (no gradients)
        predictions = self.model(images, training=False)
        
        # Calculate loss
        loss = self.loss_fn(labels, predictions)
        
        # Update metrics
        self.val_loss.update_state(loss)
        self.val_accuracy.update_state(labels, predictions)
        
        return loss.numpy()
    
    def train(self, 
             epochs: int = 10, 
             save_best_only: bool = True,
             early_stopping_patience: int = 5,
             lr_reduction_patience: int = 2,
             lr_reduction_factor: float = 0.5):
        """
        Train the model for the specified number of epochs.
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        save_best_only : bool
            Whether to save only the best model
        early_stopping_patience : int
            Number of epochs with no improvement after which training will be stopped
        lr_reduction_patience : int
            Number of epochs with no improvement after which learning rate will be reduced
        lr_reduction_factor : float
            Factor by which the learning rate will be reduced
            
        Returns:
        --------
        Dict
            Training history containing loss and accuracy values
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Initialize best validation metrics
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        lr_epochs_no_improve = 0
        
        # Initialize history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            # Reset metrics at the start of each epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            
            # Training loop
            logger.info(f"Epoch {epoch+1}/{epochs}")
            start_time = time.time()
            
            # Iterate over the training dataset
            for images, labels in self.train_dataset:
                self.train_step(images, labels)
            
            # Iterate over the validation dataset
            for images, labels in self.val_dataset:
                self.val_step(images, labels)
            
            # Get current metrics
            train_loss = self.train_loss.result().numpy()
            train_accuracy = self.train_accuracy.result().numpy()
            val_loss = self.val_loss.result().numpy()
            val_accuracy = self.val_accuracy.result().numpy()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Log metrics
            logger.info(
                f"train_loss: {train_loss:.4f}, "
                f"train_accuracy: {train_accuracy:.4f}, "
                f"val_loss: {val_loss:.4f}, "
                f"val_accuracy: {val_accuracy:.4f}, "
                f"time: {time.time() - start_time:.2f}s"
            )
            
            # Write to TensorBoard
            with self.tensorboard_writer.as_default():
                tf.summary.scalar('train_loss', train_loss, step=epoch)
                tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)
                tf.summary.scalar('val_loss', val_loss, step=epoch)
                tf.summary.scalar('val_accuracy', val_accuracy, step=epoch)
                tf.summary.scalar('learning_rate', self.optimizer.learning_rate.numpy(), step=epoch)
            
            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                lr_epochs_no_improve = 0
                
                # Save model if this is the best one
                if save_best_only:
                    logger.info(f"New best model! Saving checkpoint")
                    self.model.save_model(self.checkpoint_dir)
            else:
                epochs_no_improve += 1
                lr_epochs_no_improve += 1
                
                # Early stopping
                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Learning rate reduction
                if lr_epochs_no_improve >= lr_reduction_patience:
                    new_lr = self.optimizer.learning_rate * lr_reduction_factor
                    self.optimizer.learning_rate.assign(new_lr)
                    logger.info(f"Reducing learning rate to {new_lr}")
                    lr_epochs_no_improve = 0
            
            # Save model every epoch if not save_best_only
            if not save_best_only:
                logger.info(f"Saving checkpoint for epoch {epoch+1}")
                self.model.save_model(self.checkpoint_dir / f"epoch_{epoch+1}")
        
        logger.info(f"Training completed. Best epoch: {best_epoch+1}")
        return history
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.
        
        Parameters:
        -----------
        test_dataset : tf.data.Dataset
            Test dataset
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing test loss and accuracy
        """
        logger.info("Evaluating model on test dataset")
        
        # Initialize metrics
        test_loss = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        # Evaluate on test dataset
        for images, labels in test_dataset:
            # Forward pass
            predictions = self.model(images, training=False)
            
            # Calculate loss
            loss = self.loss_fn(labels, predictions)
            
            # Update metrics
            test_loss.update_state(loss)
            test_accuracy.update_state(labels, predictions)
        
        # Get final metrics
        test_loss_value = test_loss.result().numpy()
        test_accuracy_value = test_accuracy.result().numpy()
        
        logger.info(f"Test loss: {test_loss_value:.4f}, Test accuracy: {test_accuracy_value:.4f}")
        
        return {
            'test_loss': test_loss_value,
            'test_accuracy': test_accuracy_value
        }