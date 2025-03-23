"""
1. Loading and preprocessing data
2. Creating and training models
3. Evaluating model performance 
4. Loading the best saved model
5. Visualizing results
"""
import yaml
import tensorflow as tf
from pathlib import Path

# Import project modules
from src.utils import setup_logger, data_distribution
from src.data.data_loader import AlzheimerDataset
from src.models import SimpleCNN, DeepCNN, create_transfer_model
from src.training import Trainer
from src.evaluation import (
    calculate_metrics_from_dataset,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    plot_class_metrics,
    visualize_predictions
)

# Setup logger
logger = setup_logger('main')

def load_config(config_path):
    """Load configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return None

def create_model(model_config):
    """Create a model based on configuration."""
    input_shape = tuple(model_config['input_shape'])
    num_classes = model_config['num_classes']
    model_type = model_config['model_type']
    
    logger.info(f"Creating {model_type} model")
    
    if model_type == 'SimpleCNN':
        model = SimpleCNN(
            input_shape=input_shape,
            num_classes=num_classes
        )
    elif model_type == 'DeepCNN':
        model = DeepCNN(
            input_shape=input_shape,
            num_classes=num_classes,
            l2_rate=model_config['regularization']['l2_rate']
        )
    else:  # TransferLearningModel
        model = create_transfer_model(
            model_name=model_config['base_model'],
            num_classes=num_classes,
            input_shape=input_shape,
            freeze_base=model_config['freeze_base'],
            dropout_rate=model_config['regularization']['dropout_rate']
        )
    
    return model

def find_latest_checkpoint(model_name, checkpoint_dir='./checkpoints'):
    """Find the most recent checkpoint for a model."""
    checkpoint_path = Path(checkpoint_dir) / model_name
    
    if not checkpoint_path.exists():
        logger.warning(f"No checkpoint directory found for {model_name}")
        return None
    
    # Look for saved model files/directories
    candidates = list(checkpoint_path.glob('*'))
    
    if not candidates:
        logger.warning(f"No checkpoints found for {model_name}")
        return None
    
    # Sort by modification time (most recent first)
    latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    logger.info(f"Found latest checkpoint: {latest}")
    
    return latest

def build_model_architecture(model, input_shape):
    """Build model architecture by passing a dummy input through it."""
    logger.info("Building model architecture with dummy input")
    batch_size = 1
    dummy_input = tf.zeros([batch_size] + list(input_shape))
    _ = model(dummy_input, training=False)
    model.summary()  # Print model architecture for verification
    return model

def main():
    """Main function."""
    # Load configurations
    logger.info("Loading configurations")
    model_config = load_config('config/model_config.yaml')
    data_config = load_config('config/data_config.yaml')
    
    if not model_config or not data_config:
        logger.error("Failed to load configurations. Exiting.")
        return

    # =============================================================
    # Step 1: Load and explore data
    # =============================================================
    logger.info("Step 1: Loading and exploring data")
    
    # Visualize data distribution
    data_path = Path(data_config['data_dir'])
    class_counts = data_distribution(data_path=str(data_path))
    
    # Create an instance of AlzheimerDataset
    dataset = AlzheimerDataset(
        data_dir=data_config['data_dir'],
        img_height=data_config['preprocessing']['img_height'],
        img_width=data_config['preprocessing']['img_width'],
        batch_size=data_config['dataloader']['batch_size'],
        validation_split=data_config['split']['validation_split'],
        test_split=data_config['split']['test_split']
    )
    
    # Call create_dataset() method on the instance
    datasets = dataset.create_dataset(
        augment=data_config.get('augmentation', {}).get('enabled', True)
    )
    
    # Extract the datasets and class names
    train_ds = datasets['train']
    val_ds = datasets['validation']
    test_ds = datasets['test']
    class_names = dataset.class_names
    
    logger.info(f"Dataset loaded with classes: {class_names}")

    # =============================================================
    # Step 2: Create and train model
    # =============================================================
    logger.info("Step 2: Creating and training model")
    
    # Create model
    model = create_model(model_config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        class_names=class_names,
        learning_rate=model_config['training']['initial_learning_rate']
    )
    
    # Prepare balanced dataset
    logger.info("Preparing balanced training data")
    trainer.prepare_training_data(
        batch_size=data_config['dataloader']['batch_size']
    )
    
    # Train model
    history = trainer.train(
        epochs=model_config['training']['epochs'],
        save_best_only=model_config['training']['checkpointing']['save_best_only'],
        early_stopping_patience=model_config['training']['early_stopping']['patience'],
        lr_reduction_patience=model_config['training']['scheduler']['patience'],
        lr_reduction_factor=model_config['training']['scheduler']['factor']
    )
    
    # Plot training history
    logger.info("Plotting training history")
    plot_training_history(history)

    # =============================================================
    # Step 3: Evaluate model
    # =============================================================
    logger.info("Step 3: Evaluating model")
    
    # Calculate metrics
    metrics = calculate_metrics_from_dataset(model, test_ds, class_names)
    
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test precision: {metrics['precision']:.4f}")
    logger.info(f"Test recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 score: {metrics['f1']:.4f}")
    
    # Plot confusion matrix
    logger.info("Generating confusion matrix")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)
    
    # Plot ROC curves
    logger.info("Generating ROC curves")
    plot_roc_curves(metrics['roc_curves'], metrics['aucs'], class_names)
    
    # Plot class metrics
    logger.info("Generating class metrics")
    plot_class_metrics({
        'class_precision': metrics['class_precision'],
        'class_recall': metrics['class_recall'],
        'class_f1': metrics['class_f1']
    }, class_names)
    
    # Visualize predictions
    logger.info("Visualizing model predictions")
    visualize_predictions(model, test_ds, class_names, num_images=10)

    # =============================================================
    # Step 4: Demonstrate loading a saved model
    # =============================================================
    logger.info("Step 4: Demonstrating model loading")
    
    # Find and load the latest checkpoint
    checkpoint_path = find_latest_checkpoint(model.model_name)
    
    if checkpoint_path:
        logger.info(f"Loading model from {checkpoint_path}")
        loaded_model = create_model(model_config)
        
        # Build the model architecture before loading weights
        input_shape = tuple(model_config['input_shape'])
        loaded_model = build_model_architecture(loaded_model, input_shape)
        
        # Now load the weights after the model is built
        loaded_model.load_model(str(checkpoint_path))
        
        # Test the loaded model
        logger.info("Testing loaded model")
        loaded_metrics = calculate_metrics_from_dataset(loaded_model, test_ds, class_names)
        logger.info(f"Loaded model accuracy: {loaded_metrics['accuracy']:.4f}")
    else:
        logger.warning("No saved model found to load.")

if __name__ == "__main__":
    main()