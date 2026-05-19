"""
Flask Backend for VF Value Synthesis

Quick-start Flask server for local inference integration.

Usage:
    1. Install: pip install flask flask-cors
    2. Run: python app_template.py
    3. Server will be available at http://localhost:5000
    4. Update js/app.js to point to http://localhost:5000/api/predict

IMPORTANT:
    - This template is provided for development/research use
    - For production deployment, implement proper authentication, logging, and error handling
    - Store the model securely and do not commit it to version control
"""

import os
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile

# Import inference module
from inference import infer_single_vf
from utils import log_prediction

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_PATH = "./best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Model path: {CHECKPOINT_PATH}")

if not Path(CHECKPOINT_PATH).exists():
    print(f"[WARNING] Model file not found at {CHECKPOINT_PATH}")

# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# ============================================================================
# Helper Functions
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns:
        JSON with status information
    """
    return jsonify({
        'status': 'ok',
        'device': str(DEVICE),
        'model_exists': Path(CHECKPOINT_PATH).exists()
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expects multipart form data:
        - image: Fundus image file (required)
        - age: Age in years (optional)
        - gender: "M" or "F" (optional)
        - iop: IOP in mmHg (optional)
        - cct: CCT in micrometers (optional)
        - steps: Diffusion steps, default 300 (optional)
    
    Returns:
        JSON with VF prediction results or error message
    """
    try:
        # Check model exists
        if not Path(CHECKPOINT_PATH).exists():
            return jsonify({'error': f'Model not found at {CHECKPOINT_PATH}'}), 500
        
        # Validate image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(image_file.filename):
            return jsonify({'error': 'File must be JPG or PNG'}), 400
        
        # Extract optional clinical parameters
        age = None
        gender = None
        iop = None
        cct = None
        n_steps = 300
        
        try:
            if 'age' in request.form and request.form['age']:
                age = float(request.form['age'])
            if 'gender' in request.form and request.form['gender']:
                gender = str(request.form['gender'])
            if 'iop' in request.form and request.form['iop']:
                iop = float(request.form['iop'])
            if 'cct' in request.form and request.form['cct']:
                cct = float(request.form['cct'])
            if 'steps' in request.form and request.form['steps']:
                n_steps = int(request.form['steps'])
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid parameter format: {e}'}), 400
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_image_path = tmp.name
            image_file.save(temp_image_path)
        
        try:
            print(f"[PREDICT] Running inference:")
            print(f"  - Image: {image_file.filename}")
            print(f"  - Age: {age}")
            print(f"  - Gender: {gender}")
            print(f"  - IOP: {iop}")
            print(f"  - CCT: {cct}")
            print(f"  - Steps: {n_steps}")
            print(f"  - Device: {DEVICE}")
            
            # Run inference
            vf_denorm, vf_zscore = infer_single_vf(
                temp_image_path,
                CHECKPOINT_PATH,
                age=age,
                gender=gender,
                iop=iop,
                cct=cct,
                n_steps=n_steps,
                device=DEVICE
            )
            
            print(f"[PREDICT] Inference complete")
            print(f"  - VF mean: {np.mean(vf_denorm):.2f} dB")
            print(f"  - VF range: [{np.min(vf_denorm):.2f}, {np.max(vf_denorm):.2f}] dB")
            
            # Log to CSV
            output_id = log_prediction(
                age=age,
                gender=gender,
                iop=iop,
                cct=cct,
                vf_array=vf_denorm
            )
            
            print(f"[PREDICT] Logged to CSV: {output_id}")
            
            # Prepare response
            response = {
                'success': True,
                'vf': vf_denorm.tolist(),
                'vf_mean': float(np.mean(vf_denorm)),
                'vf_min': float(np.min(vf_denorm)),
                'vf_max': float(np.max(vf_denorm)),
                'vf_std': float(np.std(vf_denorm)),
                'model_output_id': output_id,
                'age': age,
                'gender': gender,
                'iop': iop,
                'cct': cct,
                'steps': n_steps
            }
            
            return jsonify(response), 200
        
        finally:
            # Clean up temporary file
            if Path(temp_image_path).exists():
                os.remove(temp_image_path)
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """
    Feedback submission endpoint (optional).
    
    Expects JSON:
        {
            "type": "bug|usability|feature|clinical|other",
            "name": "...",
            "email": "...",
            "message": "...",
            "rating": "excellent|good|neutral|poor",
            "consent": true/false
        }
    
    Returns:
        JSON with success/error
    """
    try:
        feedback_data = request.get_json()
        
        # Log feedback to file
        feedback_dir = Path('records')
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_log = feedback_dir / 'feedback_log.json'
        
        import json
        from datetime import datetime
        
        feedback_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            **feedback_data
        }
        
        # Append to JSONL file
        with open(feedback_log, 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        print(f"[FEEDBACK] Recorded: type={feedback_data.get('type')}, rating={feedback_data.get('rating')}")
        
        return jsonify({'success': True, 'message': 'Feedback recorded'}), 200
    
    except Exception as e:
        print(f"[ERROR] Feedback submission failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Server error'}), 500


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("VF Value Synthesis - Flask Backend")
    print("="*70)
    print(f"Starting server on http://localhost:5000")
    print(f"Device: {DEVICE}")
    print(f"Model: {CHECKPOINT_PATH}")
    print("\nEndpoints:")
    print("  GET  /api/health           - Health check")
    print("  POST /api/predict          - Run prediction")
    print("  POST /api/feedback         - Submit feedback")
    print("\nUpdate js/app.js to point to http://localhost:5000/api/predict")
    print("="*70 + "\n")
    
    # Start server
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False  # Set to False if running in production
    )
