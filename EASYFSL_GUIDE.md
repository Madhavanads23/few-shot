# Easy Few-Shot Learning Integration Guide

Your project has been successfully integrated with the **Easy Few-Shot Learning (easyfsl)** library - a professional-grade, actively maintained few-shot learning framework.

## 📦 What You Got

1. **`utils/easyfsl_integration.py`** - Wrapper class for easy FSL integration
2. **`train_easyfsl.py`** - Training script using the library
3. **`quickstart_easyfsl.py`** - Quick start examples and testing
4. **Updated requirements** - easyfsl added to dependencies

## 🚀 Quick Start

### Option 1: Use the Wrapper (Recommended)

```python
from utils.easyfsl_integration import EasyFSLWrapper
import torch.optim as optim

# Initialize model
model = EasyFSLWrapper(method='prototypical', backbone='resnet12')
optimizer = optim.Adam(model.backbone.parameters(), lr=1e-4)

# Training loop
for episode in range(num_episodes):
    # Your episodic data loading...
    loss = model.train_step(support_imgs, support_labels, 
                           query_imgs, query_labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save checkpoint
    if episode % 100 == 0:
        model.save_model('checkpoints/best_model.pt')

# Inference
predictions, probabilities = model.predict(
    support_images, support_labels, query_images
)
```

### Option 2: Use the Full Training Script

```bash
python train_easyfsl.py
```

This trains a prototypical network on your dataset with:
- Automatic learning rate scheduling
- Best model checkpointing
- Training history tracking

## 📊 Available Few-Shot Methods

### 1. **Prototypical Networks** (Recommended) ⭐
- **Speed:** Fast ⚡
- **Accuracy:** Baseline, very good
- **Complexity:** Simple
- **Best For:** Getting started, baseline comparison
- **Key Idea:** Each class is represented by the mean (prototype) of its support samples in feature space

```python
model = EasyFSLWrapper(method='prototypical')
```

### 2. **Matching Networks**
- **Speed:** Medium
- **Accuracy:** Good
- **Complexity:** Medium
- **Best For:** When attention mechanism is important
- **Key Idea:** Uses attention over support set to classify queries

```python
model = EasyFSLWrapper(method='matching')
```

### 3. **Relation Networks**
- **Speed:** Slowest
- **Accuracy:** Best potential
- **Complexity:** Most complex
- **Best For:** When you have enough training data
- **Key Idea:** Learns a learnable metric/relation module

```python
model = EasyFSLWrapper(method='relation')
```

## 🏗️ Backbone Options

- **`resnet12`** (Recommended) - Good balance of speed and accuracy
- **`resnet18`** - Lighter, faster
- **`resnet34`** - More powerful, slower

```python
model = EasyFSLWrapper(
    method='prototypical',
    backbone='resnet12',  # or 'resnet18', 'resnet34'
    num_ways=5,
    num_shots=5
)
```

## 💡 Advanced Usage

### Using Different Methods with Your Data

```python
from utils.easyfsl_integration import EasyFSLWrapper
import torch.optim as optim

# Test different methods
methods = ['prototypical', 'matching', 'relation']

for method in methods:
    model = EasyFSLWrapper(method=method, backbone='resnet12')
    optimizer = optim.Adam(model.backbone.parameters(), lr=1e-4)
    
    # Your training loop...
    for epoch in range(num_epochs):
        # Train
        loss = model.train_step(...)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate
        pred, _ = model.predict(...)
        
    model.save_model(f'checkpoints/best_{method}.pt')
```

### Loading a Saved Model

```python
model = EasyFSLWrapper(method='prototypical')
model.load_model('checkpoints/best_model.pt')

# Use for inference
predictions, probabilities = model.predict(support, labels, test)
```

## 📈 Training Tips

1. **Start with Prototypical Networks** - Simple, fast, good baseline
2. **Use ResNet12** - Industry standard for few-shot learning
3. **Learning rate scheduling** - Use cosine annealing (default in `train_easyfsl.py`)
4. **Data augmentation** - The library includes helpful transforms in `DataTransforms`
5. **Monitor validation accuracy** - Check your progress regularly

## 🎯 Recommended Hyperparameters

For CIFAR-10 or similar datasets:

```python
# Few-shot settings
n_way = 5              # 5 classes per episode
k_shot = 5             # 5 support samples per class
n_query = 5            # Query samples per class

# Training
learning_rate = 1e-4   # or 5e-4
num_epochs = 100
num_train_episodes = 300
num_val_episodes = 50

# Optimization
use_lr_scheduler = True
scheduler_type = 'cosine'  # or 'step'
```

## 🔗 Resources

- **GitHub:** https://github.com/sicara/easy-few-shot-learning
- **Documentation:** https://easyfsl.readthedocs.io/
- **Paper References:**
  - Prototypical Networks: Snell et al., 2017
  - Matching Networks: Vinyals et al., 2016
  - Relation Networks: Sung et al., 2018

## ✅ Next Steps

1. **Organize your data** - Place CIFAR-10 images in `dataset/train/` and `dataset/test/`
2. **Test with quickstart** - Run `python quickstart_easyfsl.py`
3. **Train a model** - Run `python train_easyfsl.py`
4. **Experiment with methods** - Try different methods and backbones
5. **Use for inference** - Load your best model and deploy

## 📝 File Structure

```
your_project/
├── utils/
│   ├── easyfsl_integration.py   ← Wrapper class
│   └── config.py
├── train_easyfsl.py             ← Training script
├── quickstart_easyfsl.py        ← Quick start examples
├── dataset/
│   ├── train/
│   │   ├── class1/
│   │   └── class2/
│   └── test/
│       ├── class1/
│       └── class2/
└── checkpoints/
    └── best_model.pt
```

## 🐛 Troubleshooting

**Q: ImportError for easyfsl?**
A: Run `pip install easyfsl`

**Q: CUDA out of memory?**
A: Use CPU mode or reduce batch sizes (num_shots, num_ways)

**Q: Low accuracy?**
A: Increase training episodes, adjust learning rate, use better backbones

**Q: Model not improving?**
A: Check your data format, verify episodic sampler, increase num_epochs

## 🎓 For Students/Researchers

This library is perfect for:
- **Research papers** - Compare multiple few-shot methods easily
- **Coursework** - Learn how different FSL algorithms work
- **Production** - Professional-grade, maintained code
- **Benchmarking** - Standard benchmarks and datasets

Good luck with your few-shot learning project! 🚀
