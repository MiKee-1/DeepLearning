# 🧠 Deep Learning for Abalone Age Classification

This project focuses on building a neural network model to predict the age category of abalones (Young, Middle-aged, Old) based on physical measurements. The original dataset is highly imbalanced, with very few samples in the "Old" category, making this a challenging classification task.
📊 Dataset Overview

    Original Dataset: dataset.csv

        Imbalanced classes: ~2,000 samples for "Young" and "Middle-aged", only 36 for "Old".

    Augmented Dataset: Shuffled_Dataset.csv

        Created using a custom dataset_augmentation function to synthetically balance class distribution and improve model generalization.

# 🧪 Model Architecture & Techniques

    Neural Network: Custom feedforward neural network.

    Loss Function: Focal Loss to address class imbalance.

    Regularization: L2 regularization to prevent overfitting.

    Optimizer: Adam.

    Normalization: Batch Normalization layers to stabilize training.

# ✅ Performance Metrics (Best So Far)

    Precision: 0.86

    Recall: 0.80

    F1-Score: 0.82

    Loss: 0.04

# 🚀 Goals

    Improve classification performance, especially for the minority class ("Old").

    Explore additional data augmentation and sampling strategies.

    Experiment with alternative architectures and loss functions.
