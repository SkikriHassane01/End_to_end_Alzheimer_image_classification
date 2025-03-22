import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from ..utils.logger import setup_logger

logger = setup_logger(name='src_evaluation_metrics')

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted'
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate various classification metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels (one-hot encoded)
    y_pred : np.ndarray
        Predicted probabilities
    class_names : List[str] or None
        List of class names
    average : str
        Averaging method for multi-class metrics
        
    Returns:
    --------
    Dict
        Dictionary containing various metrics
    """
    # Convert one-hot encoded to class indices
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true
    
    # Get predicted class indices
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    # Number of classes
    n_classes = y_pred.shape[1]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    precision = precision_score(y_true_indices, y_pred_indices, average=average)
    recall = recall_score(y_true_indices, y_pred_indices, average=average)
    f1 = f1_score(y_true_indices, y_pred_indices, average=average)
    
    # Calculate class-wise metrics
    class_precision = precision_score(y_true_indices, y_pred_indices, average=None)
    class_recall = recall_score(y_true_indices, y_pred_indices, average=None)
    class_f1 = f1_score(y_true_indices, y_pred_indices, average=None)
    
    # Calculate ROC curve and AUC for each class
    roc_curves = []
    aucs = []
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_curves.append((fpr, tpr))
        aucs.append(auc(fpr, tpr))
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'confusion_matrix': cm,
        'roc_curves': roc_curves,
        'aucs': aucs
    }
    
    # Add class names if provided
    if class_names is not None:
        metrics['class_names'] = class_names
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision (weighted): {precision:.4f}")
    logger.info(f"Recall (weighted): {recall:.4f}")
    logger.info(f"F1 Score (weighted): {f1:.4f}")
    
    if class_names is not None:
        for i, name in enumerate(class_names):
            logger.info(f"Class {name} - Precision: {class_precision[i]:.4f}, "
                       f"Recall: {class_recall[i]:.4f}, F1: {class_f1[i]:.4f}")
    
    return metrics

def calculate_metrics_from_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate metrics from a TensorFlow dataset.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    dataset : tf.data.Dataset
        Dataset for evaluation
    class_names : List[str] or None
        List of class names
        
    Returns:
    --------
    Dict
        Dictionary containing various metrics
    """
    # Predict on dataset
    y_true_list = []
    y_pred_list = []
    
    for images, labels in dataset:
        # Make predictions
        predictions = model(images, training=False)
        
        # Apply softmax if predictions are logits
        predictions = tf.nn.softmax(predictions)
        
        # Append to lists
        y_true_list.append(labels.numpy())
        y_pred_list.append(predictions.numpy())
    
    # Concatenate batches
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Calculate metrics
    return calculate_classification_metrics(y_true, y_pred, class_names)

def create_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    n_samples: int = 5
) -> Dict:
    """
    Create a summary of correct and incorrect predictions for analysis.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels (one-hot encoded)
    y_pred : np.ndarray
        Predicted probabilities
    class_names : List[str]
        List of class names
    n_samples : int
        Number of samples to include in each category
        
    Returns:
    --------
    Dict
        Dictionary containing sample indices for analysis
    """
    # Convert one-hot encoded to class indices
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    # Find correctly and incorrectly classified samples
    correct_indices = np.where(y_true_indices == y_pred_indices)[0]
    incorrect_indices = np.where(y_true_indices != y_pred_indices)[0]
    
    # Create dictionary for each class
    class_samples = {}
    
    for i, class_name in enumerate(class_names):
        # Samples of this class
        class_indices = np.where(y_true_indices == i)[0]
        
        # Correctly classified samples of this class
        correct_class_indices = np.intersect1d(correct_indices, class_indices)
        
        # Incorrectly classified samples of this class
        incorrect_class_indices = np.intersect1d(incorrect_indices, class_indices)
        
        # Get predicted classes for incorrect samples
        if len(incorrect_class_indices) > 0:
            incorrect_predictions = {
                class_names[pred]: [] for pred in set(y_pred_indices[incorrect_class_indices])
            }
            
            for idx in incorrect_class_indices:
                pred_class = class_names[y_pred_indices[idx]]
                incorrect_predictions[pred_class].append(idx)
        else:
            incorrect_predictions = {}
        
        class_samples[class_name] = {
            'correct': correct_class_indices[:n_samples].tolist(),
            'incorrect': {
                pred_class: indices[:n_samples] for pred_class, indices in incorrect_predictions.items()
            }
        }
    
    return class_samples