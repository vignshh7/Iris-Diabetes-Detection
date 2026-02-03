"""
Configuration file for the Diabetes Detection from Iris Images project.
This file contains all the folder paths and settings used across the project.
"""

import os

# Base directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Dataset directories (new clean structure)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
DATA_DIR = os.path.join(DATASET_DIR, 'data')
CONTROL_DIR = os.path.join(DATA_DIR, 'control')
DIABETIC_DIR = os.path.join(DATA_DIR, 'diabetic')
MASKS_DIR = os.path.join(DATASET_DIR, 'masks')
PANCREATIC_MASKS_DIR = os.path.join(DATASET_DIR, 'pancreatic_masks')

# Separate mask directories
CONTROL_MASKS_DIR = os.path.join(MASKS_DIR, 'control')
DIABETIC_MASKS_DIR = os.path.join(MASKS_DIR, 'diabetic')
CONTROL_PANCREATIC_MASKS_DIR = os.path.join(PANCREATIC_MASKS_DIR, 'control')
DIABETIC_PANCREATIC_MASKS_DIR = os.path.join(PANCREATIC_MASKS_DIR, 'diabetic')

# Pancreatic masks directories
PANCREAS_MASKS_TRAIN_DIR = os.path.join(DATASET_DIR, 'pancreas_masks_for_training')
PANCREAS_CONTROL_DIR = os.path.join(PANCREAS_MASKS_TRAIN_DIR, 'control')
PANCREAS_DIABETIC_DIR = os.path.join(PANCREAS_MASKS_TRAIN_DIR, 'diabetic')

PANCREAS_MASKS_TEST_DIR = os.path.join(DATASET_DIR, 'pancreas_masks_for_testing')
PANCREAS_TESTING_DIR = os.path.join(PANCREAS_MASKS_TEST_DIR, 'testing')
PANCREAS_TESTING1_DIR = os.path.join(PANCREAS_MASKS_TEST_DIR, 'testing1')

# Test results directories (these will be gitignored)
TEST_RESULTS_MASKS_DIR = os.path.join(PROJECT_ROOT, 'test_results_masks')
TEST_RESULTS_CONTROL_MASKS = os.path.join(TEST_RESULTS_MASKS_DIR, 'control')
TEST_RESULTS_DIABETIC_MASKS = os.path.join(TEST_RESULTS_MASKS_DIR, 'diabetic')
TEST_RESULTS_TESTING_MASKS = os.path.join(TEST_RESULTS_MASKS_DIR, 'testing')
TEST_RESULTS_TESTING1_MASKS = os.path.join(TEST_RESULTS_MASKS_DIR, 'testing1')

# Performance analysis directories (these will be gitignored)
PERFORMANCE_DIR = os.path.join(PROJECT_ROOT, 'performance_analysis')
SAMPLE_RESULTS_DIR = os.path.join(PERFORMANCE_DIR, 'sample_results')
SAMPLE_IMAGES_DIR = os.path.join(SAMPLE_RESULTS_DIR, 'images')

# Model file paths
IRIS_MODEL_2CLASS = os.path.join(MODELS_DIR, 'best_iris_model_2class.pth')
IRIS_MODEL_3CLASS = os.path.join(MODELS_DIR, 'best_iris_model_3class.pth')

FOLD_MODELS = {
    1: os.path.join(MODELS_DIR, 'best_f1_model_fold_1.pth'),
    2: os.path.join(MODELS_DIR, 'best_f1_model_fold_2.pth'),
    3: os.path.join(MODELS_DIR, 'best_f1_model_fold_3.pth'),
    4: os.path.join(MODELS_DIR, 'best_f1_model_fold_4.pth'),
    5: os.path.join(MODELS_DIR, 'best_f1_model_fold_5.pth')
}

# Result files
ANNOTATIONS_FILE = os.path.join(PROJECT_ROOT, 'annotations.csv')
EVALUATION_RESULTS = os.path.join(RESULTS_DIR, 'evaluation_results.csv')
PREDICTION_RESULTS = os.path.join(RESULTS_DIR, 'prediction_results.csv')
CROSS_VALIDATION_CHART = os.path.join(RESULTS_DIR, 'cross_validation_chart.json')

# Training parameters
IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
NUM_FOLDS = 5

# Device configuration
DEVICE = "cuda" if __name__ == "__main__" else "cpu"  # Will be set properly in actual usage

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        MODELS_DIR, DOCS_DIR, RESULTS_DIR, DATASET_DIR, CONTROL_DIR, DIABETIC_DIR,
        TESTING_DIR, TESTING1_DIR, MASKS_DIR, IMAGES_FOR_MASKS_TRAINING_DIR,
        PANCREAS_MASKS_TRAIN_DIR, PANCREAS_CONTROL_DIR, PANCREAS_DIABETIC_DIR, 
        PANCREAS_MASKS_TEST_DIR, PANCREAS_TESTING_DIR, PANCREAS_TESTING1_DIR, 
        TEST_RESULTS_MASKS_DIR, TEST_RESULTS_CONTROL_MASKS, TEST_RESULTS_DIABETIC_MASKS,
        TEST_RESULTS_TESTING_MASKS, TEST_RESULTS_TESTING1_MASKS,
        PERFORMANCE_DIR, SAMPLE_RESULTS_DIR, SAMPLE_IMAGES_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ All necessary directories created successfully!")

if __name__ == "__main__":
    create_directories()