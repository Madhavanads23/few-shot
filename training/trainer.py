import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
import os
import time

class FewShotTrainer:
    """Trainer for few-shot learning."""
    
    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: Prototypical Network
            train_loader: episodic data loader for training
            val_loader: episodic data loader for validation
            config: configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                  else 'cpu')
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 5e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Add learning rate scheduler
        if config.get('use_lr_scheduler', True):
            scheduler_type = config.get('scheduler_type', 'cosine')
            num_epochs = config.get('num_epochs', 100)
            
            if scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_epochs,
                    eta_min=1e-5
                )
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=30,
                    gamma=0.1
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.best_accuracy = 0
        self.best_model_path = config.get('best_model_path', 
                                         'best_model.pt')
    
    def train_epoch(self):
        """Train for one epoch with explicit accuracy logging."""
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        
        num_episodes = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc="Training", total=num_episodes)
        
        episode_losses = []
        episode_accs = []
        
        for episode_idx, (support_img, support_lbl, query_img, 
                         query_lbl) in enumerate(pbar):
            
            support_img = support_img.to(self.device)
            support_lbl = support_lbl.to(self.device)
            query_img = query_img.to(self.device)
            query_lbl = query_lbl.to(self.device)
            
            # Forward pass
            n_way = len(torch.unique(support_lbl))
            k_shot = self.config.get('k_shot', 5)
            logits = self.model(support_img, support_lbl, 
                               query_img, n_way, k_shot=k_shot)
            
            # Compute loss
            loss = self.criterion(logits, query_lbl)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            loss_val = loss.item()
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_lbl).float().mean().item()
            
            train_loss += loss_val
            train_accuracy += accuracy
            episode_losses.append(loss_val)
            episode_accs.append(accuracy)
            
            # Update progress bar with explicit accuracy
            avg_loss = sum(episode_losses[-10:]) / len(episode_losses[-10:])
            avg_acc = sum(episode_accs[-10:]) / len(episode_accs[-10:])
            pbar.set_postfix(OrderedDict([
                ('loss', f'{avg_loss:.4f}'),
                ('acc', f'{avg_acc:.2%}'),
                ('episode', f'{episode_idx+1}/{num_episodes}')
            ]))
        
        # Calculate averages
        avg_train_loss = train_loss / num_episodes
        avg_train_accuracy = train_accuracy / num_episodes
        
        print(f"  Train: loss={avg_train_loss:.4f}, acc={avg_train_accuracy:.2%} ({avg_train_accuracy*100:.2f}%)")
        
        return avg_train_loss, avg_train_accuracy
    
    def validate(self):
        """Validate on validation set with explicit accuracy logging."""
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        
        num_episodes = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", total=num_episodes)
            
            episode_losses = []
            episode_accs = []
            
            for support_img, support_lbl, query_img, query_lbl in pbar:
                
                support_img = support_img.to(self.device)
                support_lbl = support_lbl.to(self.device)
                query_img = query_img.to(self.device)
                query_lbl = query_lbl.to(self.device)
                
                # Forward pass
                n_way = len(torch.unique(support_lbl))
                k_shot = self.config.get('k_shot', 5)
                logits = self.model(support_img, support_lbl, 
                                   query_img, n_way, k_shot=k_shot)
                
                # Compute loss
                loss = self.criterion(logits, query_lbl)
                
                # Metrics
                loss_val = loss.item()
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_lbl).float().mean().item()
                
                val_loss += loss_val
                val_accuracy += accuracy
                episode_losses.append(loss_val)
                episode_accs.append(accuracy)
                
                # Update progress bar with explicit accuracy
                avg_loss = sum(episode_losses[-10:]) / len(episode_losses[-10:])
                avg_acc = sum(episode_accs[-10:]) / len(episode_accs[-10:])
                pbar.set_postfix(OrderedDict([
                    ('loss', f'{avg_loss:.4f}'),
                    ('acc', f'{avg_acc:.2%}')
                ]))
        
        # Calculate averages
        avg_val_loss = val_loss / num_episodes
        avg_val_accuracy = val_accuracy / num_episodes
        
        print(f"  Val:   loss={avg_val_loss:.4f}, acc={avg_val_accuracy:.2%} ({avg_val_accuracy*100:.2f}%)")
        
        return avg_val_loss, avg_val_accuracy
    
    def train(self, num_epochs):
        """Train for multiple epochs."""
        print(f"\n{'='*70}")
        print(f"TRAINING START - {num_epochs} EPOCHS")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            # Calculate remaining time
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = num_epochs - (epoch + 1)
            remaining_time = remaining_epochs * avg_epoch_time
            
            # Format time
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}h {minutes}m"
                return f"{minutes}m {secs}s"
            
            # Print nicely formatted results
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1:3d}/{num_epochs}")
            print(f"{'='*70}")
            print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:6.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}  |  Val Acc:   {val_acc*100:6.2f}%")
            
            # Learning rate
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Learning Rate: {current_lr:.2e}")
                self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(epoch, val_acc)
                print(f"  ✓✓✓ NEW BEST MODEL! Accuracy: {val_acc*100:.2f}%")
            
            # Timing info
            total_elapsed = time.time() - start_time
            print(f"  Time: {format_time(epoch_time)} (Epoch)  |  {format_time(total_elapsed)} (Total)  |  ~{format_time(remaining_time)} (Remaining)")
            
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"✓ Best Accuracy Achieved: {self.best_accuracy*100:.2f}%")
        print(f"✓ Model saved at: {self.best_model_path}")
        total_time = time.time() - start_time
        print(f"✓ Total Training Time: {format_time(total_time)}")
        print(f"{'='*70}\n")
    
    def _save_checkpoint(self, epoch, accuracy):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy
        }
        torch.save(checkpoint, self.best_model_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, 
                               map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path}")