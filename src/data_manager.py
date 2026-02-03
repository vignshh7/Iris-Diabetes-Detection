"""
Data Management Module for Diabetes Detection from Iris Images

This module provides clean, leakage-free data handling with proper train/validation/test splits.
Ensures no data leakage by maintaining strict separation between datasets.
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, List, Dict, Any
import re
import json
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class DataManager:
    """
    Centralized data management with strict train/val/test separation
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
        """
        Initialize data manager with proper split ratios
        
        Args:
            test_size: Proportion of data for final testing (never seen during training)
            val_size: Proportion of remaining data for validation
            random_state: Fixed seed for reproducible splits
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Data storage
        self.all_patients = []
        self.train_patients = []
        self.val_patients = []
        self.test_patients = []
        
        # Split info for reproducibility
        self.split_info = {}
        
        print(f"DataManager initialized:")
        print(f"  - Test split: {test_size:.1%} (held-out for final evaluation)")
        print(f"  - Validation split: {val_size:.1%} (for model selection)")
        print(f"  - Training split: {1-test_size-val_size:.1%} (for model training)")
        print(f"  - Random seed: {random_state}")
    
    def load_all_patient_data(self) -> List[Dict[str, Any]]:
        """
        Load and pair patient data from control and diabetic directories
        
        Returns:
            List of patient dictionaries with image paths and labels
        """
        print("\n=== Loading Patient Data ===")
        
        patients = []
        
        # Process Control patients (label = 0)
        control_patients = self._load_patients_from_directory(
            CONTROL_DIR, 
            PANCREAS_CONTROL_DIR,
            label=0, 
            label_name="Control"
        )
        patients.extend(control_patients)
        
        # Process Diabetic patients (label = 1) 
        diabetic_patients = self._load_patients_from_directory(
            DIABETIC_DIR,
            PANCREAS_DIABETIC_DIR, 
            label=1,
            label_name="Diabetic"
        )
        patients.extend(diabetic_patients)
        
        print(f"\nTotal patients loaded: {len(patients)}")
        print(f"  - Control patients: {len(control_patients)}")
        print(f"  - Diabetic patients: {len(diabetic_patients)}")
        
        self.all_patients = patients
        return patients
    
    def _load_patients_from_directory(self, image_dir: str, mask_dir: str, label: int, label_name: str) -> List[Dict[str, Any]]:
        """Load patient pairs from a specific directory"""
        if not os.path.exists(image_dir):
            print(f"Warning: Directory not found: {image_dir}")
            return []
        
        print(f"Loading {label_name} patients from: {image_dir}")
        
        # Group files by patient ID with improved extraction
        patient_files = defaultdict(lambda: {'L': None, 'R': None})
        
        for image_path in glob.glob(os.path.join(image_dir, '*.jpg')):
            filename = os.path.basename(image_path)
            # More robust patient ID extraction
            match = re.search(r'(\d+)(L|R)', filename, re.IGNORECASE)
            if match:
                patient_id = f"{label_name}_{match.group(1)}"  # Prefix with class to ensure uniqueness
                eye = match.group(2).upper()
                patient_files[patient_id][eye] = image_path
            else:
                print(f"  Warning: Could not extract patient ID from {filename}")
        
        # Create patient records with complete L/R pairs only
        patients = []
        for patient_id, eyes in patient_files.items():
            if eyes['L'] and eyes['R']:
                patient_record = {
                    'patient_id': patient_id,
                    'left_image': eyes['L'], 
                    'right_image': eyes['R'],
                    'left_mask': self._get_mask_path(eyes['L'], mask_dir),
                    'right_mask': self._get_mask_path(eyes['R'], mask_dir),
                    'label': label,
                    'label_name': label_name
                }
                patients.append(patient_record)
            # Skip incomplete pairs silently
        
        print(f"  Found {len(patients)} complete pairs")
        return patients
    
    def _get_mask_path(self, image_path: str, mask_dir: str) -> str:
        """Generate corresponding mask path for an image"""
        image_name = os.path.basename(image_path)
        base_name, _ = os.path.splitext(image_name)
        mask_name = f"{base_name}_pancreas_roi.png"
        return os.path.join(mask_dir, mask_name)
    
    def create_train_val_test_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create proper train/validation/test splits with no data leakage
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if not self.all_patients:
            raise ValueError("No patient data loaded. Call load_all_patient_data() first.")
        
        print(f"\n=== Creating Train/Val/Test Splits ===")
        
        # Extract patient IDs and labels for stratification
        patient_ids = [p['patient_id'] for p in self.all_patients]
        labels = [p['label'] for p in self.all_patients]
        
        print(f"Total patients: {len(patient_ids)}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # First split: separate test set (final holdout)
        train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
            range(len(self.all_patients)),
            labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state
        )
        
        # Second split: separate train and validation from remaining data
        adjusted_val_size = self.val_size / (1 - self.test_size)  # Adjust for reduced dataset
        train_indices, val_indices, _, _ = train_test_split(
            train_val_indices,
            [labels[i] for i in train_val_indices],
            test_size=adjusted_val_size,
            stratify=[labels[i] for i in train_val_indices],
            random_state=self.random_state
        )
        
        # Create data subsets
        self.train_patients = [self.all_patients[i] for i in train_indices]
        self.val_patients = [self.all_patients[i] for i in val_indices] 
        self.test_patients = [self.all_patients[i] for i in test_indices]
        
        # Ensure no duplicates within each split
        train_ids = [p['patient_id'] for p in self.train_patients]
        val_ids = [p['patient_id'] for p in self.val_patients]
        test_ids = [p['patient_id'] for p in self.test_patients]
        
        if len(set(train_ids)) != len(train_ids):
            raise ValueError(f"Duplicate patient IDs found in training set: {len(train_ids)} vs {len(set(train_ids))}")
        if len(set(val_ids)) != len(val_ids):
            raise ValueError(f"Duplicate patient IDs found in validation set: {len(val_ids)} vs {len(set(val_ids))}")
        if len(set(test_ids)) != len(test_ids):
            raise ValueError(f"Duplicate patient IDs found in test set: {len(test_ids)} vs {len(set(test_ids))}")
        
        # Store split information for reproducibility
        self.split_info = {
            'total_patients': len(self.all_patients),
            'train_indices': train_indices if isinstance(train_indices, list) else train_indices.tolist(),
            'val_indices': val_indices if isinstance(val_indices, list) else val_indices.tolist(), 
            'test_indices': test_indices if isinstance(test_indices, list) else test_indices.tolist(),
            'train_size': len(self.train_patients),
            'val_size': len(self.val_patients),
            'test_size': len(self.test_patients),
            'random_state': self.random_state,
            'split_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Print split summary
        self._print_split_summary()
        
        return self.train_patients, self.val_patients, self.test_patients
    
    def _print_split_summary(self):
        """Print detailed split summary"""
        print(f"\n--- Split Summary ---")
        for split_name, data in [("Train", self.train_patients), ("Validation", self.val_patients), ("Test", self.test_patients)]:
            labels = [p['label'] for p in data]
            label_counts = np.bincount(labels, minlength=2)
            print(f"{split_name:>10}: {len(data):3d} patients ({label_counts[0]} Control, {label_counts[1]} Diabetic)")
        
        print(f"\n--- Data Leakage Check ---")
        train_ids = set(p['patient_id'] for p in self.train_patients)
        val_ids = set(p['patient_id'] for p in self.val_patients) 
        test_ids = set(p['patient_id'] for p in self.test_patients)
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("❌ DATA LEAKAGE DETECTED!")
            if train_val_overlap: print(f"   Train-Val overlap: {train_val_overlap}")
            if train_test_overlap: print(f"   Train-Test overlap: {train_test_overlap}")
            if val_test_overlap: print(f"   Val-Test overlap: {val_test_overlap}")
            raise ValueError("CRITICAL ERROR: Data leakage detected between splits! This would invalidate all results.")
        else:
            print("✅ No data leakage detected - all splits are disjoint")
    
    def get_kfold_splits(self, n_splits: int = 5) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Create K-fold cross-validation splits from training data only
        
        Args:
            n_splits: Number of folds
            
        Returns:
            List of (train_fold, val_fold) tuples
        """
        if not self.train_patients:
            raise ValueError("No training data available. Call create_train_val_test_split() first.")
        
        print(f"\n=== Creating {n_splits}-Fold CV Splits from Training Data ===")
        
        # Extract data for stratification
        train_labels = [p['label'] for p in self.train_patients]
        
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        folds = []
        
        for fold_idx, (train_fold_indices, val_fold_indices) in enumerate(kfold.split(self.train_patients, train_labels)):
            train_fold = [self.train_patients[i] for i in train_fold_indices]
            val_fold = [self.train_patients[i] for i in val_fold_indices]
            
            folds.append((train_fold, val_fold))
            
            # Print fold summary
            train_fold_labels = np.bincount([p['label'] for p in train_fold], minlength=2)
            val_fold_labels = np.bincount([p['label'] for p in val_fold], minlength=2)
            print(f"Fold {fold_idx+1}: Train={len(train_fold)} ({train_fold_labels[0]}C,{train_fold_labels[1]}D), "
                  f"Val={len(val_fold)} ({val_fold_labels[0]}C,{val_fold_labels[1]}D)")
        
        return folds
    
    def save_split_info(self, filepath: str = "data_split_info.json"):
        """Save split information for reproducibility"""
        if not self.split_info:
            print("No split information to save. Create splits first.")
            return
        
        # Add patient IDs for each split
        detailed_info = self.split_info.copy()
        detailed_info.update({
            'train_patient_ids': [p['patient_id'] for p in self.train_patients],
            'val_patient_ids': [p['patient_id'] for p in self.val_patients],
            'test_patient_ids': [p['patient_id'] for p in self.test_patients]
        })
        
        with open(filepath, 'w') as f:
            json.dump(detailed_info, f, indent=2)
        
        print(f"Split information saved to: {filepath}")
    
    def get_train_data(self) -> List[Dict[str, Any]]:
        """Get training data"""
        return self.train_patients
    
    def get_val_data(self) -> List[Dict[str, Any]]:
        """Get validation data"""  
        return self.val_patients
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Get test data (for final evaluation only!)"""
        return self.test_patients


def create_data_manager(test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42) -> DataManager:
    """
    Factory function to create and initialize a DataManager
    
    Args:
        test_size: Proportion for test set (final holdout)
        val_size: Proportion for validation set 
        random_state: Random seed for reproducibility
        
    Returns:
        Initialized DataManager instance
    """
    dm = DataManager(test_size=test_size, val_size=val_size, random_state=random_state)
    dm.load_all_patient_data()
    dm.create_train_val_test_split()
    return dm


if __name__ == "__main__":
    # Example usage
    print("=== Data Management Module Test ===")
    
    # Create data manager with default splits
    data_manager = create_data_manager()
    
    # Save split information
    data_manager.save_split_info()
    
    # Demo K-fold splits
    kfold_splits = data_manager.get_kfold_splits(n_splits=5)
    
    print(f"\nCreated {len(kfold_splits)} K-fold splits for cross-validation")
    print("Data management setup complete!")