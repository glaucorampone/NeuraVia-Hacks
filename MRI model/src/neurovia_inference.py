"""
NeuraVia Inference Module for Alzheimer's Disease Prediction
Developed for NeuraVia Hacks Challenge
"""

import os
import sys
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Fix for TensorFlow compatibility
tf.compat.v1.disable_eager_execution = lambda: None

# Visualization imports for CAM generation
try:
    from vis.visualization import visualize_saliency, visualize_cam
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Visualization tools not available. CAM generation will be skipped.")
    VISUALIZATION_AVAILABLE = False

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
K.set_image_data_format('channels_first')

class NeuraViaPredictor:
    """
    Predictor for Alzheimer's disease classification from MRI scans
    """
    
    def __init__(self, model_path):

        # Load model with compatibility fixes
        self.model = self._load_model_with_fixes(model_path)
        self.model_name = os.path.basename(model_path)
        print(f"Model loaded: {self.model_name}")
    
    def _load_model_with_fixes(self, model_path):
        """Load model with TensorFlow compatibility fixes"""
        try:
            # First try: load without compile to avoid optimizer issues
            model = load_model(model_path, compile=False)
            
            # Recompile with modern optimizer
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            print("Model loaded and recompiled successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
           
    
    def preprocess_mri(self, nifti_path, intensity_percentile=99.0):
        """
        Preprocess MRI scan for NeuraVia prediction
        
        Args:
            nifti_path: Path to NIfTI MRI file
            intensity_percentile: Percentile for intensity normalization
            
        Returns:
            Preprocessed MRI volume ready for prediction
        """
        # Load NIfTI image
        nib_img = nib.load(nifti_path)
        volume = nib_img.get_fdata().astype('float32')
        
        # Extract brain region (standard cropping for ADNI-like data)
        brain_volume = volume[11:171, 13:205, :160]
        
        # Intensity normalization
        brain_volume = brain_volume / np.percentile(brain_volume, intensity_percentile)
        brain_volume[brain_volume > 1.0] = 1.0
        
        # Add batch and channel dimensions
        processed_volume = np.expand_dims(np.expand_dims(brain_volume, axis=0), axis=1)
        
        return processed_volume, nib_img
    
    def predict_ad_risk(self, nifti_path):
        """
        Predict Alzheimer's disease risk from MRI scan
        
        Args:
            nifti_path: Path to MRI NIfTI file
            
        Returns:
            AD risk score (0-1, higher = more likely AD)
        """
        processed_volume, _ = self.preprocess_mri(nifti_path)
        
        # Use model.predict with explicit batch processing
        prediction = self.model(processed_volume, training=False)
        ad_score = float(prediction.numpy()[0, 0])
        
        return ad_score
    
    def generate_activation_map(self, nifti_path, output_dir, 
                               generate_saliency=False, generate_cam=True):
        """
        Generate class activation maps for interpretability
        
        Args:
            nifti_path: Path to input MRI
            output_dir: Directory to save activation maps
            generate_saliency: Generate saliency maps
            generate_cam: Generate class activation maps
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization tools not available. Skipping CAM generation.")
            return
        
        processed_volume, original_nib = self.preprocess_mri(nifti_path)
        scan_id = os.path.basename(nifti_path).replace('.nii.gz', '').replace('.nii', '')
        
        layer_idx = -1  # Last layer for CAM
        
        if generate_saliency:
            for modifier in ['guided', 'relu']:
                saliency_map = visualize_saliency(
                    self.model, layer_idx, filter_indices=0,
                    seed_input=processed_volume, backprop_modifier=modifier,
                    grad_modifier='relu'
                )
                self._save_activation_map(
                    saliency_map, original_nib, output_dir,
                    f"{scan_id}_neurovia_saliency_{modifier}.nii.gz"
                )
        
        if generate_cam:
            cam_map = visualize_cam(
                self.model, layer_idx, filter_indices=0,
                seed_input=processed_volume, backprop_modifier=None,
                grad_modifier='relu'
            )
            self._save_activation_map(
                cam_map, original_nib, output_dir,
                f"{scan_id}_neurovia_cam.nii.gz"
            )
    
    def _save_activation_map(self, activation_map, original_nib, output_dir, filename):
        """Save activation map as NIfTI file"""
        # Create full-size volume
        full_volume = np.zeros(original_nib.get_fdata().shape)
        full_volume[11:171, 13:205, :160] = activation_map
        
        # Save as NIfTI
        activation_nib = nib.Nifti1Image(full_volume, original_nib.affine)
        os.makedirs(output_dir, exist_ok=True)
        nib.save(activation_nib, os.path.join(output_dir, filename))

def main():
    """Main inference function for NeuraVia prediction"""
    if len(sys.argv) != 2:
        print("Usage: python neurovia_inference.py <MODEL_PATH>")
        print("Example: python neurovia_inference.py models/neurovia_general_classifier.h5")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Initialize predictor
    predictor = NeuraViaPredictor(model_path)
    
    # Configure data paths
    data_dir = r"C:\Users\glauc\Desktop\MRI_scans"
    #cam_output_dir = "<CAM_OUTPUT_DIR>"  # Configure with output path
    
    # Find all NIfTI files
    nifti_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not nifti_files:
        nifti_files = sorted(glob.glob(os.path.join(data_dir, "*.nii")))
    
    print(f"Found {len(nifti_files)} MRI scans to process")
    
    # Process each MRI scan
    results = []
    for nifti_path in nifti_files:
        scan_id = os.path.basename(nifti_path)
        
        try:
            # Predict AD risk
            ad_score = predictor.predict_ad_risk(nifti_path)
            
            # Log result
            result_line = f"{scan_id},{predictor.model_name},{ad_score:.6f}"
            print(result_line)
            results.append(result_line)
            
            # Generate activation maps (optional)
            # predictor.generate_activation_map(nifti_path, cam_output_dir)
            
        except Exception as e:
            print(f"Error processing {scan_id}: {str(e)}")
    
    # Save results
    results_file = f"neurovia_predictions_{predictor.model_name.replace('.h5', '')}.csv"
    with open(results_file, 'w') as f:
        f.write("scan_id,model,ad_score\n")
        f.write("\n".join(results))
    
    print(f"\nNeuraVia analysis complete! Results saved to {results_file}")

if __name__ == "__main__":
    main()
