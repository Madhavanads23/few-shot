import torch
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.prototypical_network import PrototypicalNetwork
from data.data_loader import EpisodicDataLoader
from utils.config import Config
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================================
# GLOBAL MODEL & DATA LOADER
# ============================================================

model = None
data_loader = None
device = None
config = None

def load_model_and_data():
    """Load model and data loader on startup."""
    global model, data_loader, device, config
    
    print("Loading model and data...")
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = PrototypicalNetwork(
        backbone_type=config.backbone_type,
        feature_dim=config.feature_dim,
        num_transformer_layers=config.num_transformer_layers,
        similarity=config.similarity_metric
    ).to(device)
    
    # Load checkpoint (with fallback for mismatched architectures)
    if os.path.exists(config.best_model_path):
        try:
            checkpoint = torch.load(config.best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded from {config.best_model_path}")
        except (RuntimeError, KeyError) as e:
            print(f"⚠ Checkpoint format mismatch: {str(e)[:100]}...")
            print("  Using untrained model (training in Colab will save compatible checkpoint)")
    else:
        print("⚠ No checkpoint found - using untrained model")
    
    model.eval()
    
    # Load data loader for support images
    data_loader = EpisodicDataLoader(
        root_dir=config.dataset_dir,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        split='test',
        image_size=config.image_size
    )
    
    print("✓ Model and data loaded successfully")

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(image_path, n_way=5):
    """
    Classify uploaded image using few-shot learning.
    
    Args:
        image_path: path to uploaded image
        n_way: number of ways (classes)
    
    Returns:
        results: dict with predictions and confidence
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply same transforms as data loader
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    query_image = transform(image).unsqueeze(0)  # (1, 3, H, W)
    
    # Generate support set
    support_img, support_lbl = [], []
    
    with torch.no_grad():
        for i in range(n_way):
            # Get k_shot random samples from each class
            class_list = list(data_loader.class_images.keys())
            if i < len(class_list):
                class_name = class_list[i]
                class_images = data_loader.class_images[class_name]
                
                for _ in range(config.k_shot):
                    img_path = np.random.choice(class_images)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    support_img.append(img_tensor)
                    support_lbl.append(i)
        
        support_images = torch.stack(support_img).to(device)
        support_labels = torch.tensor(support_lbl, dtype=torch.long).to(device)
        query_image = query_image.to(device)
        
        # Predict
        logits = model(support_images, support_labels, query_image, n_way, 
                      k_shot=config.k_shot)
        
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        
        # Get class name
        class_list = list(data_loader.class_images.keys())
        if predicted.item() < len(class_list):
            predicted_class = class_list[predicted.item()]
        else:
            predicted_class = f"Class {predicted.item()}"
        
        results = {
            'predicted_class': predicted_class,
            'predicted_idx': predicted.item(),
            'confidence': confidence[0].item(),
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'classes': class_list[:n_way]
        }
        
        return results

def create_prediction_chart(probabilities, classes):
    """Create matplotlib chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['#2ecc71' if i == np.argmax(probabilities) else '#3498db' 
              for i in range(len(classes))]
    ax.bar(classes, probabilities, color=colors)
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Classification Confidence by Class', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    
    # Add percentage labels on bars
    for i, (cls, prob) in enumerate(zip(classes, probabilities)):
        ax.text(i, prob + 0.02, f'{prob*100:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to base64 for embedding in HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    """Home page."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
    
    return render_template('index.html', 
                          gpu_available=gpu_available,
                          gpu_name=gpu_name)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and classification."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Only {ALLOWED_EXTENSIONS} files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Classify
        print(f"\nClassifying: {filename}")
        results = classify_image(filepath, n_way=config.n_way)
        
        # Create chart
        chart_image = create_prediction_chart(results['probabilities'], 
                                             results['classes'])
        
        results['chart'] = chart_image
        results['filename'] = filename
        results['success'] = True
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/uploads/<filename>')
def download_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/info')
def info():
    """Get model info."""
    return jsonify({
        'model': config.backbone_type,
        'n_way': config.n_way,
        'k_shot': config.k_shot,
        'feature_dim': config.feature_dim,
        'image_size': config.image_size,
        'device': str(device)
    })

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    load_model_and_data()
    
    print("\n" + "=" * 60)
    print("STARTING FLASK APP")
    print("=" * 60)
    print("Open browser: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
