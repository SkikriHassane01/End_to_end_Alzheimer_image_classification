```
alzheimers-detection/
├── data/
│   ├── raw/                    # Original downloaded images
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/val/test splits
├── notebooks/
│   ├── 1-exploratory-analysis.ipynb
│   ├── 2-data-preprocessing.ipynb
│   ├── 3-baseline-models.ipynb
│   ├── 4-advanced-cnn-models.ipynb
│   └── 5-model-evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Dataset loading utilities
│   │   ├── augmentation.py     # Data augmentation strategies
│   │   └── preprocessing.py    # MRI preprocessing functions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py       # Abstract model class
│   │   ├── cnn_models.py       # Custom CNN architectures
│   │   ├── transfer_models.py  # Transfer learning models
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop utilities
│   │   └── class_balancing.py  # Imbalance handling strategies
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Classification metrics
│   │   └── eval_visualization.py    # Results visualization
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # Logging configuration
│       └── visualization.py    # MRI visualization tools
├── configs/
│   ├── data_config.yaml        # Data configuration
│   └── model_config.yaml       # Model hyperparameters
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── setup.py