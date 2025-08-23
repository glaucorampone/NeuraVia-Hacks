#!/usr/bin/env python3
"""
Alzheimer MRI Classification Inference
Dataset: Falah/Alzheimer_MRI
Model: DHEIVER/Alzheimer-MRI
"""

import os
import warnings
import json
from typing import Dict, Any, Tuple

# Suppress verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt


def load_model_and_processor(model_name: str) -> Tuple[Any, Any, str]:
    """Load model and processor with device detection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor, device


def run_inference(model: Any, processor: Any, image: Any, device: str) -> Dict[str, Any]:
    """Run inference on input image."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_class_idx = predictions.argmax().item()
    confidence = predictions.max().item()
    probabilities = predictions.cpu().numpy()[0]
    
    # Get class label from model config or use index
    if hasattr(model.config, 'id2label') and model.config.id2label:
        predicted_label = model.config.id2label[predicted_class_idx]
        all_labels = [model.config.id2label[i] for i in range(len(probabilities))]
    else:
        predicted_label = f"Class_{predicted_class_idx}"
        all_labels = [f"Class_{i}" for i in range(len(probabilities))]
    
    return {
        "predicted_label": predicted_label,
        "predicted_class_idx": predicted_class_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "all_labels": all_labels
    }


def save_results(image: Any, results: Dict[str, Any], model_name: str, 
                original_label: Any, device: str, output_dir: str = "outputs") -> None:
    """Save visualization and JSON results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original MRI Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Probability chart
    probs = results["probabilities"]
    labels = results["all_labels"]
    predicted_idx = results["predicted_class_idx"]
    
    colors = ['red' if i == predicted_idx else 'lightblue' for i in range(len(probs))]
    bars = axes[1].bar(range(len(probs)), probs, color=colors)
    
    axes[1].set_title(f"Classification: {results['predicted_label']}\n"
                     f"Confidence: {results['confidence']:.3f}", 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Classes", fontweight='bold')
    axes[1].set_ylabel("Probability", fontweight='bold')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Add probability values on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    image_path = f"{output_dir}/classification_result.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save JSON results
    results_data = {
        "model": model_name,
        "predicted_class": results["predicted_label"],
        "predicted_class_idx": results["predicted_class_idx"],
        "confidence": float(results["confidence"]),
        "all_probabilities": results["probabilities"].tolist(),
        "original_label": str(original_label),
        "image_size": list(image.size),
        "device_used": device
    }
    
    json_path = f"{output_dir}/results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)


def main():
    """Main inference pipeline."""
    # Configuration
    MODEL_NAME = "DHEIVER/Alzheimer-MRI"
    DATASET_NAME = "Falah/Alzheimer_MRI"
    SAMPLE_INDEX = 0
    
    # Load dataset and select sample
    dataset = load_dataset(DATASET_NAME, split="train")
    sample = dataset[SAMPLE_INDEX]
    image = sample['image'].convert('RGB')
    original_label = sample.get('label', 'N/A')
    
    # Load model and processor
    model, processor, device = load_model_and_processor(MODEL_NAME)
    
    # Run inference
    results = run_inference(model, processor, image, device)
    
    # Save results
    save_results(image, results, MODEL_NAME, original_label, device)
    
    return results


if __name__ == "__main__":
    results = main()
