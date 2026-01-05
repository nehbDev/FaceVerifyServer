from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import requests
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

def process_image_input(image_input):
    """Process image from URL, base64, or file upload"""
    try:
        # Handle file upload
        if hasattr(image_input, 'read'):
            image_data = image_input.read()
            image = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Handle URL
        elif isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
            response = requests.get(image_input, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Handle base64
        else:
            if ',' in image_input:
                image_input = image_input.split(',')[1]
            
            image_data = base64.b64decode(image_input)
            image = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")

def get_embedding_fast(img):
    """Fast face embedding extraction with optimized settings"""
    try:
        embedding = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="ssd",
            enforce_detection=False,
            align=False
        )
        return embedding[0]["embedding"]
    except Exception as e:
        raise Exception(f"Failed to extract face embedding: {str(e)}")

def verify_faces_fast(embedding1, embedding2, threshold=0.50):
    """Fast face verification using cosine similarity"""
    try:
        emb1 = np.array(embedding1, dtype=np.float32)
        emb2 = np.array(embedding2, dtype=np.float32)
        
        cos_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        
        distance = 1 - cos_sim
        verified = distance < threshold
        
        return {
            "verified": bool(verified),
            "distance": float(distance),
            "threshold": float(threshold),
            "similarity": float(cos_sim),
            "match": bool(verified)
        }
    except Exception as e:
        raise Exception(f"Verification failed: {str(e)}")

@app.route('/python/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Face verification server is running"})

@app.route('/python/verify', methods=['POST'])
def verify_images():
    try:
        content_type = request.content_type or ''
        
        if 'application/json' in content_type:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            ref_img = process_image_input(data.get('reference_image'))
            cap_img = process_image_input(data.get('captured_image'))
                
        elif 'multipart/form-data' in content_type:
            reference_file = request.files.get('reference_image')
            captured_file = request.files.get('captured_image')
            
            ref_img = process_image_input(reference_file) if reference_file else process_image_input(request.form.get('reference_image'))
            cap_img = process_image_input(captured_file) if captured_file else process_image_input(request.form.get('captured_image'))
        else:
            return jsonify({"error": "Unsupported Media Type"}), 415
        
        ref_embedding = get_embedding_fast(ref_img)
        cap_embedding = get_embedding_fast(cap_img)
        result = verify_faces_fast(ref_embedding, cap_embedding)
        
        return jsonify({"success": True, "result": result})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)