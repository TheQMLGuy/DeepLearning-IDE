# 🔥 PyTorch Deep Learning IDE - Complete Guide

## 📋 Overview

A **professional-grade PyTorch development environment** with **50+ features** designed to streamline your deep learning workflow. From data loading to deployment, everything you need is integrated and automated.

---

## 🚀 Quick Start

### Installation

```python
# The PyTorch modules are already in src/pytorch/
# They work with Pyodide (browser Python runtime)

# In a notebook cell:
from pytorch.trainer import AutoTrainer, LRFinder
from pytorch.model_builder import ModelTemplates, ModelSurgery
from pytorch.data_pipeline import SmartDataLoader, AugmentationPipeline
from pytorch.debugging import NaNDetector, GradientChecker
```

### 30-Second Training Pipeline

```python
# 1. Load data
from torchvision import datasets
transform = AugmentationPipeline.image_classification_train()
dataset = datasets.CIFAR10('./data', train=True, transform=transform)

train_loader = SmartDataLoader.create(dataset, batch_size=64)
val_loader = SmartDataLoader.create(val_dataset, batch_size=64)

# 2. Create model
model = ModelTemplates.resnet18(num_classes=10)

# 3. Train automatically
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.use_amp = True  # Mixed precision for 2-3x speedup
trainer.train(epochs=50, early_stopping_patience=10)

# Done! Model is trained, checkpointed, and logged to TensorBoard
```

---

## 📦 50+ Features Organized by Category

### 🔄 Training Features (7 features)

#### 1. **Auto Training Loop**
Never write training loops again. Handles epochs, validation, checkpointing automatically.

```python
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.train(epochs=50)
```

#### 2. **Learning Rate Finder**
Find optimal LR in seconds using Leslie Smith's method.

```python
lr_finder = LRFinder(model, optimizer, criterion)
results = lr_finder.range_test(train_loader)
optimal_lr = results['optimal_lr']  # Use this!
```

#### 3. **Mixed Precision Training**
2-3x faster training with FP16, automatically handled.

```python
trainer.use_amp = True  # That's it!
```

#### 4. **Gradient Accumulation**
Train with larger batch sizes on small GPUs.

```python
trainer.gradient_accumulation_steps = 4
# Effective batch = batch_size * 4
```

#### 5. **Smart Early Stopping**
Monitors validation loss, prevents overfitting.

```python
# Already integrated in trainer
trainer.train(epochs=100, early_stopping_patience=10)
```

#### 6. **Auto-Resume Training**
Crash recovery built-in.

```python
trainer.auto_resume = True
# Automatically resumes from last checkpoint
```

#### 7. **Curriculum Learning**
Start easy, gradually increase difficulty.

```python
trainer.curriculum_fn = lambda epoch: min(1.0, epoch/10)
```

---

### 🏗️ Architecture Features (6 features)

#### 8. **Model Templates**
Pre-built architectures: ResNet, VGG, U-Net, Transformer, LSTM, VAE, GAN.

```python
model = ModelTemplates.resnet18(num_classes=10)
model = ModelTemplates.transformer_encoder(d_model=512)
model = ModelTemplates.unet(in_channels=3, out_channels=1)
model = ModelTemplates.lstm_classifier(input_size=100, hidden_size=128)
model = ModelTemplates.vae(input_dim=784, latent_dim=20)
gan = ModelTemplates.gan(latent_dim=100)
```

#### 9. **Model Surgery - Freeze Layers**
Easy transfer learning.

```python
ModelSurgery.freeze_layers(model, until='layer3')
ModelSurgery.unfreeze_layers(model, from_layer='layer4')
```

#### 10. **Model Surgery - Replace Head**
Change output classes instantly.

```python
ModelSurgery.replace_head(model, num_classes=100)
```

#### 11. **Model Summary**
See parameters, shapes, FLOPs.

```python
ModelSummary.summary(model, input_size=(3, 224, 224))
```

#### 12. **Parameter Counter**
Count trainable vs frozen parameters.

```python
params = ModelSurgery.count_parameters(model)
print(f"Trainable: {params['trainable']:,}")
```

#### 13. **Add Dropout**
Add regularization to existing models.

```python
model = ModelSurgery.add_dropout(model, p=0.5)
```

---

### 💾 Data Pipeline Features (10 features)

#### 14. **Smart DataLoader**
Auto-optimized data loading.

```python
loader = SmartDataLoader.create(dataset, batch_size=64)
# Automatically sets: num_workers, pin_memory, persistent_workers
```

#### 15. **Augmentation Pipelines**
Pre-built augmentation for different tasks.

```python
transform = AugmentationPipeline.image_classification_train()
transform = AugmentationPipeline.strong_augmentation()
transform = AugmentationPipeline.image_classification_val()
```

#### 16. **Cutout Augmentation**
```python
cutout = AugmentationPipeline.cutout(n_holes=1, length=16)
```

#### 17. **MixUp**
State-of-the-art augmentation.

```python
mixed_x, y_a, y_b, lam = AugmentationPipeline.mixup_batch(x, y, alpha=1.0)
loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

#### 18. **Dataset Analysis**
Comprehensive stats in seconds.

```python
stats = DatasetAnalyzer.analyze(dataset)
# Shows: class distribution, balance, mean/std, corrupted samples
```

#### 19. **Find Corrupted Samples**
```python
corrupted_indices = DatasetAnalyzer.find_corrupted(dataset)
```

#### 20. **Imbalanced Data Handler**
Auto-balance classes.

```python
balanced_loader = ImbalancedDataHandler.create_balanced_loader(dataset, batch_size=64)
```

#### 21. **Augmentation Preview**
See augmentations before training.

```python
previews = AugmentationPreview.preview(dataset, transform, num_samples=9)
```

#### 22. **Data Splitter**
Smart train/val/test splits.

```python
train_ds, val_ds, test_ds = DataSplitter.split(dataset, train_ratio=0.8)
```

#### 23. **Stratified Split**
Preserves class distribution.

```python
train_ds, val_ds, test_ds = DataSplitter.stratified_split(dataset)
```

---

### 🐛 Debugging Features (8 features)

#### 24. **NaN/Inf Detector**
Automatically catches numerical instabilities.

```python
detector = NaNDetector(model)
detector.register_hooks()
# Training automatically stops on first NaN, shows where
```

#### 25. **Gradient Checker**
Verify gradients are flowing.

```python
stats = GradientChecker.check_gradients(model)
# Shows which layers have zero gradients
```

#### 26. **Gradient Flow Plot**
Visualize gradient magnitudes.

```python
flow_data = GradientChecker.plot_gradient_flow(model)
# Returns data for plotting
```

#### 27. **Memory Profiler**
Track GPU memory usage per layer.

```python
memory_stats = MemoryProfiler.profile(model, input_size=(3, 224, 224))
```

#### 28. **Speed Profiler**
Find slow layers and bottlenecks.

```python
speed_stats = SpeedProfiler.profile(model, input_size=(3, 224, 224))
```

#### 29. **Shape Tracer**
Track tensor shapes through network.

```python
shapes = ShapeTracer.trace(model, input_size=(3, 224, 224))
```

#### 30. **Backward Debugger**
Inspect gradients during backprop.

```python
debugger = BackwardDebugger(model)
debugger.register_hooks()
# ... training ...
debugger.print_stats()
```

#### 31. **Model Health Check**
Comprehensive diagnostic.

```python
report = ModelHealthCheck.check(model, sample_input)
```

---

### 💾 Checkpointing & Logging (4 features)

#### 32. **Checkpoint Manager**
Smart checkpoint saving.

```python
ckpt_manager = CheckpointManager(save_dir='./checkpoints')
ckpt_manager.save(model, optimizer, epoch, metrics, is_best=True)
```

#### 33. **Best Model Tracking**
Automatically keeps top-N best models.

```python
# Configured in CheckpointManager
# Automatically saves top 3 best models by val_loss
```

#### 34. **TensorBoard Integration**
Automatic logging.

```python
# Already integrated in AutoTrainer
# Logs: loss, accuracy, learning rate, gradients
```

#### 35. **Training History**
Saves metrics to JSON.

```python
history = trainer.train(epochs=50)
# Auto-saved to checkpoints/history.json
```

---

### ⚙️ Optimization Features (5 features)

#### 36. **Learning Rate Schedulers**
Pre-configured best practices.

```python
from torch.optim.lr_scheduler import *

trainer.scheduler = CosineAnnealingLR(optimizer, T_max=50)
trainer.scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=50, steps_per_epoch=len(train_loader))
trainer.scheduler = ReduceLROnPlateau(optimizer, patience=5)
```

#### 37. **Gradient Clipping**
Prevent gradient explosion.

```python
trainer.clip_grad_norm = 1.0
```

#### 38. **Weight Decay Scheduling**
Dynamic regularization.

```python
# Built into optimizers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

#### 39. **Optimizer Comparison**
Test multiple optimizers.

```python
# Create separate trainers for each optimizer
results = {}
for opt_class in [torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW]:
    optimizer = opt_class(model.parameters(), lr=1e-3)
    trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
    history = trainer.train(epochs=10)
    results[opt_class.__name__] = history
```

#### 40. **Automatic Batch Size Finder**
Find optimal batch size for your GPU.

```python
# Coming soon - finds max batch size that fits in memory
```

---

### 📊 Visualization Features (5+ features)

#### 41. **Live Training Dashboard**
Real-time plots during training.

```python
# Integrated with TensorBoard
# Run: tensorboard --logdir=./runs
```

#### 42. **Loss & Accuracy Plots**
```python
# Auto-saved in history
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.legend()
```

#### 43. **Learning Rate Plot**
```python
plt.plot(history['lr'])
plt.ylabel('Learning Rate')
```

#### 44. **Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
# Get predictions
preds, labels = [], []
for x, y in val_loader:
    pred = model(x).argmax(1)
    preds.extend(pred.cpu().numpy())
    labels.extend(y.cpu().numpy())

cm = confusion_matrix(labels, preds)
```

#### 45. **Feature Visualization**
```python
# Extract features
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer4.register_forward_hook(get_activation('layer4'))
output = model(image)
features = activations['layer4']
```

---

### 🚀 Deployment Features (5 features)

#### 46. **ONNX Export**
Deploy anywhere.

```python
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

#### 47. **TorchScript Conversion**
Production-ready models.

```python
scripted = torch.jit.script(model)
scripted.save('model.pt')
```

#### 48. **Quantization**
4x smaller models.

```python
import torch.quantization as quantization
model.eval()
quantized = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

#### 49. **Model Pruning**
Remove unnecessary weights.

```python
import torch.nn.utils.prune as prune

for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

#### 50. **Inference Optimization**
Fuse layers for speed.

```python
model.eval()
model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
```

---

## 🎯 Complete Workflow Examples

### Example 1: Image Classification (CIFAR-10)

```python
import torch
import torch.nn as nn
from torchvision import datasets
from pytorch.trainer import AutoTrainer, LRFinder
from pytorch.model_builder import ModelTemplates
from pytorch.data_pipeline import *

# 1. Data
transform_train = AugmentationPipeline.image_classification_train(img_size=32)
transform_val = AugmentationPipeline.image_classification_val(img_size=32)

train_dataset = datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
val_dataset = datasets.CIFAR10('./data', train=False, transform=transform_val)

# Analyze dataset
stats = DatasetAnalyzer.analyze(train_dataset)

# Create loaders
train_loader = SmartDataLoader.create(train_dataset, batch_size=128)
val_loader = SmartDataLoader.create(val_dataset, batch_size=128, shuffle=False)

# 2. Model
model = ModelTemplates.resnet18(num_classes=10, input_channels=3)

# 3. Optimizer & Criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Find optimal learning rate
lr_finder = LRFinder(model, optimizer, criterion)
results = lr_finder.range_test(train_loader)
print(f"Optimal LR: {results['optimal_lr']:.2e}")

# Update optimizer with optimal LR
for param_group in optimizer.param_groups:
    param_group['lr'] = results['optimal_lr']

# 5. Train
trainer = AutoTrainer(
    model, train_loader, val_loader, optimizer, criterion,
    save_dir='./checkpoints/cifar10',
    log_dir='./runs/cifar10'
)

# Configure training features
trainer.use_amp = True
trainer.gradient_accumulation_steps = 1
trainer.clip_grad_norm = 1.0
trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Train!
history = trainer.train(epochs=100, early_stopping_patience=15)

# 6. Evaluate
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

accuracy = 100. * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")

# 7. Export model
torch.onnx.export(model, torch.randn(1, 3, 32, 32).cuda(), 'cifar10_model.onnx')
print("✓ Model exported to ONNX")
```

### Example 2: Transfer Learning (Custom Dataset)

```python
from pytorch.model_builder import ModelTemplates, ModelSurgery

# 1. Load pretrained model
model = ModelTemplates.resnet18(num_classes=1000)  # ImageNet weights

# 2. Freeze early layers
ModelSurgery.freeze_layers(model, until='layer3')

# 3. Replace head for your classes
ModelSurgery.replace_head(model, num_classes=5)

# 4. Check parameters
params = ModelSurgery.count_parameters(model)
print(f"Trainable: {params['trainable']:,} / {params['total']:,}")

# 5. Train only the head first
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.train(epochs=10)

# 6. Unfreeze and fine-tune
ModelSurgery.unfreeze_layers(model, from_layer='layer3')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower LR
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.train(epochs=20)
```

### Example 3: Debugging a Model

```python
from pytorch.debugging import *

# 1. Health check
report = ModelHealthCheck.check(model, sample_input)

# 2. Set up NaN detector
detector = NaNDetector(model)
detector.register_hooks()

# 3. Check gradients
model.zero_grad()
output = model(sample_input)
loss = criterion(output, target)
loss.backward()

GradientChecker.check_gradients(model)

# 4. Profile memory
memory_stats = MemoryProfiler.profile(model, input_size=(3, 224, 224))

# 5. Profile speed
speed_stats = SpeedProfiler.profile(model, input_size=(3, 224, 224))

# 6. Trace shapes
shapes = ShapeTracer.trace(model, input_size=(3, 224, 224))

detector.remove_hooks()
```

---

## 📚 Best Practices

### 1. Always Use These Features:
- ✅ `AutoTrainer` for all training
- ✅ `LRFinder` to find optimal learning rate
- ✅ Mixed precision (`trainer.use_amp = True`)
- ✅ Early stopping
- ✅ TensorBoard logging
- ✅ Gradient clipping for RNNs/Transformers

### 2. Debugging Workflow:
1. `ModelHealthCheck` first
2. Enable `NaNDetector` during training
3. `GradientChecker` if training fails
4. `MemoryProfiler` if OOM errors
5. `SpeedProfiler` to optimize inference

### 3. Data Pipeline:
1. Always `analyze_dataset()` first
2. Use stratified splits for imbalanced data
3. Preview augmentations before training
4. Use `ImbalancedDataHandler` for class imbalance

### 4. Model Development:
1. Start with templates
2. Use `ModelSummary` to understand architecture
3. Freeze/unfreeze layers for transfer learning
4. Count parameters regularly

---

## 🎓 Learning Resources

### Concepts to Master:
1. **Mixed Precision Training**: 2-3x speedup with minimal accuracy loss
2. **Learning Rate Scheduling**: Critical for convergence
3. **Transfer Learning**: Start from pretrained weights
4. **Regularization**: Dropout, weight decay, data augmentation
5. **Gradient Clipping**: Essential for RNNs/Transformers

### Further Reading:
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CS231n: CNNs](http://cs231n.stanford.edu/)
- [Fast.ai Course](https://course.fast.ai/)

---

## 🤝 Contributing

This is a comprehensive toolkit - contributions welcome!

---

## 📄 License

MIT License - Use freely in your projects!

---

**Happy Deep Learning! 🔥**
