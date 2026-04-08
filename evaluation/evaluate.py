import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class FewShotEvaluator:
    """Evaluate few-shot learning model."""
    
    def __init__(self, model, data_loader, device='cuda'):
        """
        Args:
            model: PrototypicalNetwork
            data_loader: episodic data loader
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
    
    def evaluate(self, num_episodes=100, n_way=5):
        """
        Evaluate on multiple episodes.
        
        Args:
            num_episodes: number of episodes to evaluate
            n_way: number of ways
        
        Returns:
            results: dict with accuracy and other metrics
        """
        self.model.eval()
        
        accuracies = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                support_img, support_lbl, query_img, query_lbl = \
                    self.data_loader.generate_episode()
                
                support_img = support_img.to(self.device)
                support_lbl = support_lbl.to(self.device)
                query_img = query_img.to(self.device)
                query_lbl = query_lbl.to(self.device)
                
                # Predict
                predictions, confidence, logits = self.model.predict(
                    support_img, support_lbl, query_img, n_way)
                
                # Compute accuracy
                accuracy = (predictions == query_lbl).float().mean().item()
                accuracies.append(accuracy)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(query_lbl.cpu().numpy())
        
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        results = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'accuracies': accuracies,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return results
    
    def compute_confusion_matrix(self, predictions, labels, n_way):
        """Compute confusion matrix."""
        cm = confusion_matrix(labels, predictions, 
                             labels=list(range(n_way)))
        return cm
    
    def plot_confusion_matrix(self, cm, class_names=None, save_path=None):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar=True)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def print_evaluation_report(self, predictions, labels, class_names=None):
        """Print classification report."""
        report = classification_report(labels, predictions, 
                                      target_names=class_names)
        print(report)