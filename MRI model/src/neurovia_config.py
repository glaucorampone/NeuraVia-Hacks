import os

# Model Configuration
NEUROVIA_MODELS = {
    'general': 'models/neurovia_general_classifier.h5',
}

# Data Paths (Configure these for your environment)
DATA_PATHS = {
    'subjects_csv': 'dataset/neurovia_subjects.csv',
    'brain_masks': 'dataset/brain_masks/'
}

# Model Parameters
MODEL_CONFIG = {
    'input_shape': [160, 192, 160],
    'intensity_percentile': 99.0,
    'network_depth': 5,
    'base_filters': 4,
    'batch_size': 5,
    'learning_rate': 0.0001
}

# Clinical Thresholds
CLINICAL_THRESHOLDS = {
    'low_risk': 0.3,     # Below this: likely CN
    'high_risk': 0.7,    # Above this: likely AD
    # Between low_risk and high_risk: requires clinical evaluation
}

# Preprocessing Settings
PREPROCESSING = {
    'brain_crop': {
        'x_start': 11, 'x_end': 171,
        'y_start': 13, 'y_end': 205,
        'z_start': 0, 'z_end': 160
    },
    'normalization': {
        'method': 'percentile',
        'percentile': 99.0,
        'clip_max': 1.0
    }
}

def get_model_path(model_name='general'):
    """Get path to NeuraVia model"""
    return NEUROVIA_MODELS.get(model_name, NEUROVIA_MODELS['general'])

def validate_paths():
    """Validate that required paths exist"""
    issues = []
    
    # Check model files
    for name, path in NEUROVIA_MODELS.items():
        if not os.path.exists(path):
            issues.append(f"Model not found: {path}")
    
    # Check dataset files
    if not os.path.exists(DATA_PATHS['subjects_csv']):
        issues.append(f"Subjects CSV not found: {DATA_PATHS['subjects_csv']}")
    
    if not os.path.exists(DATA_PATHS['brain_masks']):
        issues.append(f"Brain masks directory not found: {DATA_PATHS['brain_masks']}")
    
    return issues

def interpret_score(ad_score):
    """Interpret AD risk score"""
    if ad_score < CLINICAL_THRESHOLDS['low_risk']:
        return "Low AD Risk (Likely Cognitively Normal)"
    elif ad_score > CLINICAL_THRESHOLDS['high_risk']:
        return "High AD Risk (Likely Alzheimer's Disease)"
    else:
        return "Moderate Risk"

if __name__ == "__main__":
    # Validate configuration
    issues = validate_paths()
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("NeuraVia configuration validated successfully!")
    
    print(f"\nAvailable models: {list(NEUROVIA_MODELS.keys())}")
    print(f"Default model: {get_model_path()}")
