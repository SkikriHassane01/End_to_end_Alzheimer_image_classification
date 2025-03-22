import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
import tensorflow as tf
from ..utils.logger import setup_logger

logger = setup_logger(name='eval_visualization')

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    confusion_matrix : np.ndarray
        Confusion matrix
    class_names : List[str]
        List of class names
    figsize : Tuple[int, int]
        Figure size
    normalize : bool
        Whether to normalize the confusion matrix
    save_path : str or None
        Path to save the figure (if None, the figure is displayed)
    """
    plt.figure(figsize=figsize)
    
    if normalize:
        # Normalize confusion matrix
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        title = "Normalized Confusion Matrix"
        fmt = '.2f'
    else:
        cm = confusion_matrix
        title = "Confusion Matrix"
        fmt = 'd'
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

def plot_roc_curves(
    roc_curves: List[Tuple[np.ndarray, np.ndarray]],
    aucs: List[float],
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curves for each class.
    
    Parameters:
    -----------
    roc_curves : List[Tuple[np.ndarray, np.ndarray]]
        List of (fpr, tpr) tuples for each class
    aucs : List[float]
        List of AUC values for each class
    class_names : List[str]
        List of class names
    figsize : Tuple[int, int]
        Figure size
    save_path : str or None
        Path to save the figure (if None, the figure is displayed)
    """
    plt.figure(figsize=figsize)
    
    # Plot each ROC curve
    for i, ((fpr, tpr), auc_value) in enumerate(zip(roc_curves, aucs)):
        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC = {auc_value:.2f})"
        )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path)
        logger.info(f"ROC curves saved to {save_path}")
    else:
        plt.show()

def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and accuracy).
    
    Parameters:
    -----------
    history : Dict[str, List[float]]
        Dictionary containing training history
    figsize : Tuple[int, int]
        Figure size
    save_path : str or None
        Path to save the figure (if None, the figure is displayed)
    """
    plt.figure(figsize=figsize)
    
    # Create subplots
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Training history saved to {save_path}")
    else:
        plt.show()

def plot_class_metrics(
    metrics: Dict[str, np.ndarray],
    class_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot class-wise metrics.
    
    Parameters:
    -----------
    metrics : Dict[str, np.ndarray]
        Dictionary containing class-wise metrics
    class_names : List[str]
        List of class names
    figsize : Tuple[int, int]
        Figure size
    save_path : str or None
        Path to save the figure (if None, the figure is displayed)
    """
    plt.figure(figsize=figsize)
    
    # Get class metrics
    precision = metrics['class_precision']
    recall = metrics['class_recall']
    f1 = metrics['class_f1']
    
    # Set up bar positions
    x = np.arange(len(class_names))
    width = 0.25
    
    # Create bars
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1')
    
    # Add labels and title
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Class-wise Performance Metrics', fontsize=16)
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim([0, 1.05])
    
    # Add value labels on top of bars
    for i, v in enumerate(precision):
        plt.text(i - width, v + 0.02, f"{v:.2f}", ha='center')
    for i, v in enumerate(recall):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    for i, v in enumerate(f1):
        plt.text(i + width, v + 0.02, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    # Save or display figure
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Class metrics saved to {save_path}")
    else:
        plt.show()

def visualize_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: List[str],
    num_images: int = 10,
    save_dir: Optional[str] = None
) -> None:
    """
    Visualize model predictions on sample images.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    dataset : tf.data.Dataset
        Dataset containing images and labels
    class_names : List[str]
        List of class names
    num_images : int
        Number of images to visualize
    save_dir : str or None
        Directory to save the figures (if None, figures are displayed)
    """
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get a batch of images from the dataset
    images_so_far = 0
    
    for images, labels in dataset:
        batch_size = images.shape[0]
        
        # Make predictions
        predictions = model(images, training=False)
        predictions = tf.nn.softmax(predictions)
        
        # Get predicted and true classes
        pred_classes = tf.argmax(predictions, axis=1).numpy()
        true_classes = tf.argmax(labels, axis=1).numpy()
        
        # Iterate through images in the batch
        for i in range(batch_size):
            if images_so_far >= num_images:
                return
            
            # Plot image
            plt.figure(figsize=(6, 6))
            plt.imshow(tf.squeeze(images[i]), cmap='gray')
            
            # Add prediction info
            pred_class = class_names[pred_classes[i]]
            true_class = class_names[true_classes[i]]
            confidence = 100 * np.max(predictions[i])
            
            title_color = 'green' if pred_classes[i] == true_classes[i] else 'red'
            plt.title(f"Pred: {pred_class} ({confidence:.1f}%)\nTrue: {true_class}", 
                    color=title_color, fontsize=14)
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save or display figure
            if save_dir:
                plt.savefig(f"{save_dir}/prediction_{images_so_far}.png")
                plt.close()
            else:
                plt.show()
            
            images_so_far += 1
            
            if images_so_far >= num_images:
                return