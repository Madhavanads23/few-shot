"""
Production CLIP Model Wrapper
==============================
Simple, optimized wrapper for pre-trained CLIP deployment

Usage:
    model = CLIPModelWrapper()
    prediction = model.predict('path/to/image.jpg')
    print(f"Class: {prediction['class']}, Confidence: {prediction['confidence']:.2%}")
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import json


class CLIPModelWrapper:
    """Production-ready CLIP model wrapper."""
    
    # CIFAR-10 class descriptions (optimized for zero-shot CLIP)
    CIFAR10_CLASSES = {
        'airplane': 'a photo of an airplane',
        'automobile': 'a photo of an automobile/car',
        'bird': 'a photo of a bird',
        'cat': 'a photo of a cat',
        'deer': 'a photo of a deer',
        'dog': 'a photo of a dog',
        'frog': 'a photo of a frog',
        'horse': 'a photo of a horse',
        'ship': 'a photo of a ship',
        'truck': 'a photo of a truck',
    }
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize CLIP model.
        
        Args:
            model_name: HuggingFace model ID
            device: torch device (None = auto-detect)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"📥 Loading pre-trained CLIP model ({model_name})...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ CLIP model loaded on {self.device}")
        
        # Precompute text embeddings
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precompute text embeddings for all classes."""
        print("📝 Precomputing text embeddings...")
        
        class_names = list(self.CIFAR10_CLASSES.keys())
        class_texts = [self.CIFAR10_CLASSES[name] for name in class_names]
        
        with torch.no_grad():
            text_inputs = self.processor(
                text=class_texts, 
                return_tensors="pt", 
                padding=True
            )
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.model.get_text_features(**text_inputs)
            self.text_embeddings = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        
        self.class_names = class_names
        print(f"✅ Text embeddings ready for {len(class_names)} classes")
    
    def predict(self, image_path, return_all_scores=False):
        """
        Predict class for a single image.
        
        Args:
            image_path: path to image file
            return_all_scores: if True, return scores for all classes
        
        Returns:
            dict with:
                - 'class': predicted class name
                - 'confidence': prediction confidence (0-1)
                - 'class_index': numeric class index
                - 'all_scores' (optional): scores for all classes
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                image_inputs = self.processor(
                    images=image, 
                    return_tensors="pt"
                )
                image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
                
                # Get image embeddings
                image_features = self.model.get_image_features(**image_inputs)
                image_embeddings = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                
                # Compute similarity scores between image and each class text
                # Simple dot product between (1, 512) and (10, 512)
                similarity_scores = (image_embeddings @ self.text_embeddings.T).squeeze(0)
                
                # Get prediction - similarity_scores is shaped (10,)
                confidence = torch.max(similarity_scores).item()
                predicted_idx = torch.argmax(similarity_scores).item()
                predicted_class = self.class_names[predicted_idx]
            
            result = {
                'class': predicted_class,
                'confidence': confidence,
                'class_index': predicted_idx,
                'device': str(self.device),
                'success': True
            }
            
            if return_all_scores:
                # Convert similarity scores (shape 10,) to numpy
                scores_np = similarity_scores.cpu().numpy()
                result['all_scores'] = {
                    name: float(score) 
                    for name, score in zip(self.class_names, scores_np)
                }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'class': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths: list of image paths
            batch_size: number of images to process at once
        
        Returns:
            list of prediction dicts
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                result = self.predict(path)
                results.append({
                    'image': str(path),
                    **result
                })
            
            print(f"  ✅ Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
        
        return results
    
    def evaluate_directory(self, data_dir, split='test'):
        """
        Evaluate model on a directory of images organized by class.
        
        Args:
            data_dir: root directory
            split: 'test' or 'train'
        
        Returns:
            dict with accuracy metrics
        """
        print(f"📂 Evaluating on {split} set...")
        
        test_dir = Path(data_dir) / split
        
        predictions = []
        ground_truth = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = test_dir / class_name
            
            if not class_dir.exists():
                continue
            
            image_files = list(class_dir.glob('*'))
            
            for img_path in image_files:
                result = self.predict(img_path)
                if result['success']:
                    predictions.append(result['class_index'])
                    ground_truth.append(class_idx)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_images': len(ground_truth),
            'correct': sum(p == g for p, g in zip(predictions, ground_truth))
        }


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("CLIP MODEL WRAPPER - DEMO")
    print("="*70)
    
    # Initialize model
    model = CLIPModelWrapper()
    
    # Test on single image
    print("\n📷 Testing on single image...")
    test_image = "dataset/test/dog/00001.png"
    
    if Path(test_image).exists():
        result = model.predict(test_image, return_all_scores=True)
        print(f"\nPrediction: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if 'all_scores' in result:
            print("\nAll class scores:")
            for class_name, score in sorted(result['all_scores'].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"  {class_name:15} : {score:.4f}")
    else:
        print(f"Image not found: {test_image}")
