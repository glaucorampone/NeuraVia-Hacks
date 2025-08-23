from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from alzheimer_dheiver_inference import load_model_and_processor, run_inference, process_image_file
from PIL import Image
import io
import torch

app = FastAPI(title="Alzheimer MRI Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "DHEIVER/Alzheimer-MRI"
model, processor, device = load_model_and_processor(MODEL_NAME)


@app.post("/process")
async def process_mri(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    results = process_image_file(image)
    results_serializable = {
        "predicted_label": str(results["predicted_label"]),
        "predicted_class_idx": int(results["predicted_class_idx"]),
        "confidence": float(results["confidence"]),
        "probabilities": list(results["probabilities"]),  # se già lista, lascia così
        "all_labels": list(results["all_labels"])
    }
    return results_serializable
