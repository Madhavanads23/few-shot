"""
Production CLIP Deployment - Flask Web App
===========================================
Simple web interface for pre-trained CLIP model

Run: python clip_deployment_app.py
Visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import os
import json
from werkzeug.utils import secure_filename
from PIL import Image
import torch

from clip_model_wrapper import CLIPModelWrapper

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Disable template caching
app.jinja_env.cache = None  # Disable Jinja2 cache

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Disable caching
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# ============================================================
# GLOBAL MODEL
# ============================================================

model = None

def load_model():
    """Load CLIP model on startup."""
    global model
    print("Loading pre-trained CLIP model...")
    model = CLIPModelWrapper()
    print("✅ Model ready for inference!")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================================
# ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def index():
    """Homepage with upload interface."""
    return render_template('clip_index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single image prediction."""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))
        
        # Make prediction
        result = model.predict(str(filepath), return_all_scores=True)
        
        # Format response
        if result['success']:
            response = {
                'success': True,
                'prediction': result['class'],
                'confidence': f"{result['confidence']:.2%}",
                'all_scores': result.get('all_scores', {})
            }
        else:
            response = {
                'success': False,
                'error': result['error']
            }
        
        # Cleanup
        try:
            filepath.unlink()
        except:
            pass
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch processing."""
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No files selected'}), 400
    
    try:
        results = []
        filepaths = []
        
        # Save files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(str(filepath))
                filepaths.append(filepath)
        
        # Run predictions
        for filepath in filepaths:
            result = model.predict(str(filepath))
            
            if result['success']:
                results.append({
                    'file': filepath.name,
                    'prediction': result['class'],
                    'confidence': f"{result['confidence']:.2%}"
                })
            else:
                results.append({
                    'file': filepath.name,
                    'error': result['error']
                })
        
        # Cleanup
        for filepath in filepaths:
            try:
                filepath.unlink()
            except:
                pass
        
        return jsonify({
            'success': True,
            'total': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get model information."""
    return jsonify({
        'model': 'OpenAI CLIP ViT-base-patch32',
        'device': str(model.device),
        'classes': list(model.CIFAR10_CLASSES.keys()),
        'num_classes': len(model.CIFAR10_CLASSES),
        'accuracy_on_cifar10': '89.66%',
        'trained': False
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_ready': model is not None
    })

# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal errors."""
    print(f"ERROR: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================
# STARTUP
# ============================================================

@app.before_request
def before_request():
    """Initialize model on first request if not already done."""
    global model
    if model is None and request.path not in ['/health']:
        load_model()

if __name__ == '__main__':
    print("="*70)
    print("🚀 CLIP MODEL DEPLOYMENT SERVER")
    print("="*70)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Model: OpenAI CLIP ViT-base-patch32")
    print(f"Accuracy on CIFAR-10: 89.66%")
    print()
    print("Starting Flask app...")
    print("✅ Service running at http://localhost:5000")
    print("="*70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
