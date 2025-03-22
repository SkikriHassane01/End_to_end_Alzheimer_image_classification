import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from .logger import setup_logger
from pathlib import Path
logger = setup_logger(name='src_utils_visualization')
from typing import Optional
DATA_PATH = Path(__file__).resolve().parents[2] / "Data" / "raw"
SAVE_PLOT_PATH = Path(__file__).resolve().parents[2] / "reports"

def data_distribution(data_path:str=DATA_PATH, save_figure:Optional[bool]=False, save_plot_path:Optional[str]=SAVE_PLOT_PATH) -> None:
    """
    Display the distribution of data in the data folder.
    
    Parameters:
    ----------
    data_path : str
        Path to the data folder
    save_figure : bool
        Save the plot as a PNG file
    save_plot_path : str
        Path to save the plot
        
    Returns:
    -------
    class_counts : dict
        Number of images in each class
    """
    try:
        if os.path.exists(data_path):
            logger.info(f"Data directory found at: {DATA_PATH}")
            
            # get the list of classes in the data folder 
            classes = [d.name for d in os.scandir(data_path) if d.is_dir()]
            n_classes = len(classes)
            logger.info(f"Found {n_classes} classes: {', '.join(classes)}")
            
            # get the number of images in each classes
            class_counts = {}
            for cls in classes:
                image_files = [f for f in os.listdir(os.path.join(data_path, cls)) if f.endswith(('.png', '.jpg', '.jpeg'))]
                class_counts[cls] = len(image_files)
            
            # Create bar plot
            bars = plt.bar(class_counts.keys(), class_counts.values(), 
                    color=sns.color_palette("viridis", len(class_counts)))
            
            # Add counts on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{height}',
                        ha='center', va='bottom', fontsize=12)
            
            plt.title("MRI Scan Distribution by Class", fontsize=16)
            plt.xlabel("Class")
            plt.ylabel("Number of Images")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
            # pie chart
            plt.figure(figsize=(8,8))
            plt.pie(
                x = class_counts.values(),
                labels=class_counts.keys(),
                autopct="%1.1f%%",
                startangle=90,
                explode=[0.05]*n_classes,
                shadow=True
            )
            plt.title("Data Distribution")
            plt.show()
            
            # save the plot if required
            if save_figure:
                plot_path = os.path.join(save_plot_path, "1_data_distribution.png")
                plt.savefig(plot_path)
                logger.info(f"Data distribution plot saved at: {plot_path}")
            
            return class_counts
        else:
            logger.error(f"Data folder not found at {data_path}")
            return None
    except Exception as e:
        logger.error(f"Error checking data folder: {e}")
        return None

def get_sample_images(classes, data_path:str=DATA_PATH, samples_per_class=4):
    """
    Get sample images from each class.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset
    classes : list
        List of class directories
    samples_per_class : int
        Number of samples to get from each class
        
    Returns:
    --------
    dict : Dictionary of class_name -> list of file paths
    """
    samples = {}
    
    for cls in classes:
        class_path = os.path.join(data_path, cls)
        
        if not os.path.exists(class_path):
            logger.warning(f"Class directory not found: {class_path}")
            continue
            
        # Get all image files
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {class_path}")
            continue
            
        # Select random samples
        import random
        sample_files = random.sample(image_files, 
                                    min(samples_per_class, len(image_files)))
        
        samples[cls] = sample_files
        
    return samples
    
def load_image(file_path):
    """
    Load a PNG image file.
    
    Parameters:
    ----------
    file_path : str
        Path to the image file
        
    Returns:
    -------
    numpy.ndarray
        Image data as numpy array
    """
    try:
        with Image.open(file_path) as img:
            # Convert to grayscale if it's a color image
            if img.mode != 'L':
                img = img.convert('L')
            
            # Convert to numpy array
            img_array = np.array(img)
            
            logger.info(f"Loaded image {file_path} with shape {img_array.shape}")
            return img_array
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        return None

def display_image(img_array, title=None):
    """
    Display a single image.
    
    Parameters:
    ----------
    img_array : numpy.ndarray
        Image data
    title : str or None
        Title for the plot
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array, cmap='gray')
    
    if title:
        plt.title(title)
    
    plt.colorbar(label='Intensity')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_multiple_images(image_files, max_images=9):
    """
    Display multiple images in a grid.
    
    Parameters:
    ----------
    image_files : list
        List of image file paths
    max_images : int
        Maximum number of images to display
    """
    # Limit the number of images
    if len(image_files) > max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"Displaying {len(image_files)} images")
    
    # Calculate grid dimensions
    n_images = len(image_files)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    # Load and display each image
    for i, file_path in enumerate(image_files):
        if i < len(axes):
            img_array = load_image(file_path)
            if img_array is not None:
                axes[i].imshow(img_array, cmap='gray')
                axes[i].set_title(os.path.basename(file_path))
                axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()