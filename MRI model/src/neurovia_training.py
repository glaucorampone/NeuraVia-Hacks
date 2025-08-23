"""
Training Module for Alzheimer's Disease Classification
"""

import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from neurovia_network import create_neurovia_model

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_neurovia_classifier(
    inference_only=False,
    brain_region="whole_brain",
    volume_shape=[160, 192, 160],
    num_classes=1,
    fine_tune=False,
    registration="affine",
    use_pretrained=True,
    network_depth=5,
    initial_filters=4,
    conv_regularization=0.0,
    dense_regularization=1.0,
    learning_rate=0.0001,
    batch_size=5,
    pretrained_path=None,
    use_global_pool=False,
    model_name_suffix="",
):
    
    # Configure training parameters
    if fine_tune:
        use_pretrained = True
        trainable = False
    else:
        trainable = True

    if use_pretrained and pretrained_path is None:
        print("ERROR: Pretrained model path required but not provided")
        return None

    # Data configuration
    if registration == "affine":
        data_dir = "<NIFTI_DATA_DIR>"  # Configure with actual data path
        subjects_df = pd.read_csv("dataset/neurovia_subjects.csv")

    # Extract data columns
    file_ids = subjects_df["file"].values
    diagnoses = subjects_df["dx"].values
    cv_splits = subjects_df["cross_validation_split_ad_cn"].values
    scan_numbers = subjects_df["scan"].values
    subject_ids = subjects_df["RID"].values

    # Filter data (exclude MCI and follow-up scans for baseline training)
    exclude_mask = np.logical_or(
        np.logical_or((diagnoses > 3), (diagnoses == 2)), 
        np.logical_or((cv_splits == -1), scan_numbers > 0)
    )
    
    valid_indices = ~exclude_mask
    file_ids = file_ids[valid_indices]
    diagnoses = diagnoses[valid_indices]
    cv_splits = cv_splits[valid_indices]
    scan_numbers = scan_numbers[valid_indices]
    subject_ids = subject_ids[valid_indices]

    # Convert diagnoses to binary (0: CN, 1: AD)
    binary_labels = np.where(diagnoses == 3, 1, 0)
    
    # Preprocessing parameters
    intensity_percentile = 99.0
    max_epochs = 100
    
    # Load brain region mask if specified
    if brain_region != "whole_brain":
        mask_path = f"dataset/brain_masks/mask_5e-3_{brain_region}_all_1496.nii.gz"
        if not os.path.exists(mask_path):
            print(f"ERROR: Mask file not found: {mask_path}")
            return None
        
        import nibabel as nib
        mask_volume = np.squeeze(
            nib.load(mask_path).get_fdata()[11:171, 13:205, :160]
        ).astype(bool)
    else:
        mask_volume = None

    # Create or load model
    if use_pretrained:
        print(f"Loading pretrained model: {pretrained_path}")
        model = load_model(pretrained_path)
    else:
        print("Creating new NeuraVia model...")
        model = create_neurovia_model(
            input_dims=volume_shape,
            output_classes=num_classes,
            depth=network_depth,
            base_filters=initial_filters,
            conv_l2=conv_regularization,
            dense_l2=dense_regularization,
            learning_rate=learning_rate,
            trainable=trainable,
            global_pool=use_global_pool
        )

    # Model summary
    print("NeuraVia Model Architecture:")
    model.summary()

    # Training callbacks
    model_filename = f"models/neurovia_classifier_{model_name_suffix}.h5"
    callbacks = [
        ModelCheckpoint(
            model_filename,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
    ]

    # Data loading and training would continue here...
    print("NeuraVia training configuration complete!")
    print(f"Model will be saved as: {model_filename}")
    
    return model

if __name__ == "__main__":
    # Example training configuration for NeuraVia Hack
    model = train_neurovia_classifier(
        brain_region="whole_brain",
        network_depth=5,
        learning_rate=0.0001,
        model_name_suffix="neurovia_hack"
    )
