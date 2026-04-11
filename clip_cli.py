"""
CLIP Model - Command Line Interface
====================================
Simple CLI tool for making predictions

Usage:
    python clip_cli.py [image_path]
    python clip_cli.py --batch [directory]
    python clip_cli.py --evaluate dataset/test
"""

import argparse
import json
from pathlib import Path
from tabulate import tabulate

from clip_model_wrapper import CLIPModelWrapper


def main():
    parser = argparse.ArgumentParser(
        description='CLIP Model Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python clip_cli.py path/to/image.jpg
  
  # Batch process directory
  python clip_cli.py --batch dataset/test/dog
  
  # Evaluate on test set (showing per-class accuracy)
  python clip_cli.py --evaluate dataset/test
  
  # Single image with all class scores
  python clip_cli.py path/to/image.jpg --scores
        """
    )
    
    parser.add_argument(
        'image', 
        nargs='?', 
        help='Path to image file'
    )
    parser.add_argument(
        '--batch', 
        action='store_true',
        help='Batch process all images in directory (use with image path)'
    )
    parser.add_argument(
        '--evaluate', 
        metavar='DIR',
        help='Evaluate on test directory (requires class subdirectories)'
    )
    parser.add_argument(
        '--scores', 
        action='store_true',
        help='Show scores for all classes'
    )
    parser.add_argument(
        '--output', 
        metavar='FILE',
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Initialize model
    print("="*70)
    print("🚀 CLIP MODEL - COMMAND LINE INTERFACE")
    print("="*70)
    print()
    
    model = CLIPModelWrapper()
    
    # Single image prediction
    if args.image and not args.batch and not args.evaluate:
        print_single_prediction(model, args.image, args.scores)
    
    # Batch processing
    elif args.batch and args.image:
        print_batch_predictions(model, args.image, args.output)
    
    # Evaluation
    elif args.evaluate:
        print_evaluation(model, args.evaluate)
    
    else:
        parser.print_help()


def print_single_prediction(model, image_path, show_scores=False):
    """Print prediction for single image."""
    
    path = Path(image_path)
    
    if not path.exists():
        print(f"❌ Error: File not found: {image_path}")
        return
    
    print(f"📷 Predicting: {image_path}")
    print()
    
    result = model.predict(image_path, return_all_scores=True)
    
    if not result['success']:
        print(f"❌ Error: {result['error']}")
        return
    
    # Print result
    print("RESULT:")
    print("-" * 70)
    print(f"  Predicted Class: {result['class']}")
    print(f"  Confidence:      {result['confidence']:.2%}")
    print(f"  Class Index:     {result['class_index']}")
    print()
    
    if show_scores and 'all_scores' in result:
        print("ALL CLASS SCORES:")
        print("-" * 70)
        
        scores = sorted(
            result['all_scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        table_data = [
            [i+1, name, f"{score:.2%}"]
            for i, (name, score) in enumerate(scores)
        ]
        
        print(tabulate(
            table_data,
            headers=['Rank', 'Class', 'Confidence'],
            tablefmt='plain'
        ))
    
    print()
    print("="*70)


def print_batch_predictions(model, directory, output_file=None):
    """Batch process all images in directory."""
    
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Error: Directory not found: {directory}")
        return
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = [
        f for f in dir_path.rglob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"❌ No images found in {directory}")
        return
    
    print(f"📂 Processing {len(image_files)} images from {directory}")
    print()
    
    results = model.predict_batch(image_files)
    
    # Print results
    print()
    print("BATCH RESULTS:")
    print("="*70)
    
    table_data = [
        [
            r['image'].split('/')[-1],
            r.get('class', 'ERROR'),
            f"{r.get('confidence', 0):.2%}",
            '✅' if r.get('success') else '❌'
        ]
        for r in results
    ]
    
    print(tabulate(
        table_data,
        headers=['Image', 'Prediction', 'Confidence', 'Status'],
        tablefmt='grid'
    ))
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print()
    print(f"✅ Successful: {successful}/{len(results)}")
    
    # Save to file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to: {output_file}")
    
    print("="*70)


def print_evaluation(model, test_dir):
    """Evaluate model on test directory."""
    
    print(f"📊 Evaluating on: {test_dir}")
    print()
    
    metrics = model.evaluate_directory(test_dir)
    
    print()
    print("EVALUATION RESULTS:")
    print("="*70)
    print(f"  Total Images: {metrics['total_images']}")
    print(f"  Correct:      {metrics['correct']}")
    print(f"  Wrong:        {metrics['total_images'] - metrics['correct']}")
    print()
    print("METRICS:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1-Score:  {metrics['f1_score']:.2%}")
    print("="*70)


if __name__ == '__main__':
    main()
