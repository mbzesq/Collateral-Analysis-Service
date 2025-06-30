import os
import tempfile
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from pdf2image import convert_from_path
import numpy as np
import sys
from pathlib import Path
from paddleocr import PaddleOCR

# Add current directory to path to import config
sys.path.append(str(Path(__file__).parent))
from config import MODEL_OUTPUT_PATH

app = Flask(__name__)

# Initialize PaddleOCR on startup
print("Initializing PaddleOCR engine...")
# This will download the necessary models on the first run
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
print("PaddleOCR engine initialized successfully.")

# Load the trained document classification model on startup
doc_model = None
try:
    doc_model = joblib.load(MODEL_OUTPUT_PATH)
    print(f"Document classification model '{MODEL_OUTPUT_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"WARNING: Document classification model file not found at '{MODEL_OUTPUT_PATH}'. The /predict endpoint will not work.")


@app.route('/')
def health_check():
    """A simple health check endpoint."""
    model_status = "loaded" if doc_model is not None else "not_loaded"
    return jsonify({
        "status": "ok", 
        "message": "NPLVision Document Classification API is running.",
        "service": "document_classification",
        "model_status": model_status
    })

@app.route('/predict', methods=['POST'])
def predict_document_type():
    """
    Accepts a PDF file and returns the predicted document type for each page,
    using parallel processing for optimization.
    """
    if doc_model is None:
        return jsonify({"error": "Document classification model not loaded. Cannot perform prediction."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # Convert PDF pages to images
        images = convert_from_path(file)

        predictions = []
        for i, img in enumerate(images):
            # Use PaddleOCR to get structured text results
            # The OCR result is a list of lists, e.g., [[...], [[...]]]
            ocr_result = ocr_engine.ocr(np.array(img), cls=True)

            # Extract just the text from the results and join it into a single string
            page_text = ""
            if ocr_result and ocr_result[0] is not None:
                text_lines = [line[1][0] for line in ocr_result[0]]
                page_text = " ".join(text_lines)

            # Use the existing scikit-learn model for classification
            prediction = doc_model.predict([page_text])
            predictions.append({
                "page": i + 1,
                "predicted_label": prediction[0]
            })
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "page_count": len(predictions),
            "predictions": predictions
        })

    except Exception as e:
        # Clean up temporary file if it exists
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"An error occurred during document prediction: {e}")
        return jsonify({"error": "Failed to process PDF", "details": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get information about the loaded document classification model.
    """
    if doc_model is None:
        return jsonify({
            "model_loaded": False,
            "error": "Document classification model not loaded"
        }), 503
    
    try:
        # Try to load model metadata if available
        metadata_path = MODEL_OUTPUT_PATH.parent / f"{MODEL_OUTPUT_PATH.stem}_metadata.json"
        metadata = {}
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            pass
        
        # Get model information
        model_info_data = {
            "model_loaded": True,
            "model_type": str(type(doc_model)),
            "model_path": str(MODEL_OUTPUT_PATH)
        }
        
        # Add metadata if available
        if metadata:
            model_info_data.update({
                "accuracy": metadata.get("accuracy"),
                "training_samples": metadata.get("training_samples"),
                "test_samples": metadata.get("test_samples"),
                "supported_labels": metadata.get("labels"),
                "min_accuracy_threshold": metadata.get("min_accuracy_threshold")
            })
        
        return jsonify(model_info_data)
        
    except Exception as e:
        return jsonify({
            "model_loaded": True,
            "error": f"Could not retrieve model information: {str(e)}"
        }), 500

@app.route('/api-info', methods=['GET'])
def api_info():
    """
    Get information about the document classification API endpoints.
    """
    model_status = "loaded" if doc_model is not None else "not_loaded"
    
    return jsonify({
        "service_name": "NPLVision Document Classification API",
        "version": "1.0",
        "model_status": model_status,
        "endpoints": {
            "/": "Health check",
            "/predict": "POST - Classify PDF document pages",
            "/model-info": "GET - Get document classification model information",
            "/api-info": "GET - Get API information"
        },
        "supported_document_types": [
            "Note", "Mortgage", "Deed of Trust", "Assignment", 
            "Allonge", "Rider", "Bailee Letter", "UNLABELED"
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)