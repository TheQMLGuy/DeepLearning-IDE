# 🔥 PyTorch IDE - Integration & Usage Guide

## 🚀 Quick Integration

### Step 1: Enable PyTorch Features in IDE

The PyTorch modules are already created in `src/pytorch/`. They work with Pyodide (browser Python).

### Step 2: Add PyTorch Panel to IDE

Modify `src/App.tsx` to add PyTorch features panel:

```typescript
import { useState } from 'react';
import { Brain, Zap, Database, Eye, Bug } from 'lucide-react';

// Add to your App component
const [showPyTorchPanel, setShowPyTorchPanel] = useState(false);

// Add button to header
<button 
  className={`toggle-btn ${showPyTorchPanel ? 'active' : ''}`}
  onClick={() => setShowPyTorchPanel(!showPyTorchPanel)}
>
  <Brain size={14} />
  <span>PyTorch</span>
</button>

// Add panel in main content area
{showPyTorchPanel && (
  <PyTorchPanel onInsertCode={insertCodeToCell} />
)}
```

### Step 3: Use in Notebooks

Simply import and use in your notebook cells:

```python
# Cell 1: Setup
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Load PyTorch IDE modules (these are in src/pytorch/)
import sys
sys.path.append('./pytorch')

from trainer import AutoTrainer, LRFinder
from model_builder import ModelTemplates, ModelSurgery
from data_pipeline import SmartDataLoader, AugmentationPipeline
from debugging import NaNDetector, GradientChecker

# Cell 2: Quick model
model = ModelTemplates.resnet18(num_classes=10)
print(model)

# Cell 3: Data
transform = AugmentationPipeline.image_classification_train()
dataset = datasets.CIFAR10('./data', train=True, transform=transform, download=True)
loader = SmartDataLoader.create(dataset, batch_size=64)

# Cell 4: Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.use_amp = True
history = trainer.train(epochs=50, early_stopping_patience=10)

# Cell 5: Evaluate
# ... your evaluation code
```

---

## 📋 Feature Cheat Sheet

### 🎯 Most Used Features

Copy-paste these snippets directly into notebook cells:

#### 1. Quick Training Setup
```python
# Complete training in 5 lines
model = ModelTemplates.resnet18(num_classes=10)
loader = SmartDataLoader.create(dataset, batch_size=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss())
history = trainer.train(epochs=50)
```

#### 2. Find Optimal Learning Rate
```python
lr_finder = LRFinder(model, optimizer, criterion)
results = lr_finder.range_test(train_loader)
print(f"Use LR: {results['optimal_lr']:.2e}")
```

#### 3. Transfer Learning
```python
model = ModelTemplates.resnet18(num_classes=1000)
ModelSurgery.freeze_layers(model, until='layer3')
ModelSurgery.replace_head(model, num_classes=YOUR_CLASSES)
```

#### 4. Debug Training Issues
```python
# If training fails:
ModelHealthCheck.check(model, sample_input)
GradientChecker.check_gradients(model)

# If NaN/Inf appears:
detector = NaNDetector(model)
detector.register_hooks()
# ... train ...
detector.remove_hooks()
```

#### 5. Analyze Dataset
```python
stats = DatasetAnalyzer.analyze(dataset)
# Shows: class distribution, balance, corrupted samples
```

#### 6. Handle Imbalanced Data
```python
balanced_loader = ImbalancedDataHandler.create_balanced_loader(
    dataset, batch_size=64
)
```

#### 7. Profile Model
```python
# Memory usage
MemoryProfiler.profile(model, input_size=(3, 224, 224))

# Speed
SpeedProfiler.profile(model, input_size=(3, 224, 224))

# Shapes
ShapeTracer.trace(model, input_size=(3, 224, 224))
```

---

## 🎨 Code Templates

### Template 1: Image Classification

```python
#%% python
"""
Complete Image Classification Pipeline
Modify: num_classes, img_size, model architecture
"""

import torch
import torch.nn as nn
from torchvision import datasets
from trainer import AutoTrainer, LRFinder
from model_builder import ModelTemplates
from data_pipeline import *

# Config
NUM_CLASSES = 10
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 100

# 1. Data
transform_train = AugmentationPipeline.image_classification_train(img_size=IMG_SIZE)
transform_val = AugmentationPipeline.image_classification_val(img_size=IMG_SIZE)

train_dataset = datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
val_dataset = datasets.CIFAR10('./data', train=False, transform=transform_val)

train_loader = SmartDataLoader.create(train_dataset, batch_size=BATCH_SIZE)
val_loader = SmartDataLoader.create(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Model
model = ModelTemplates.resnet18(num_classes=NUM_CLASSES, input_channels=3)

# 3. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Find LR
lr_finder = LRFinder(model, optimizer, criterion)
results = lr_finder.range_test(train_loader, num_iter=100)
optimal_lr = results['optimal_lr']
print(f"📈 Optimal LR: {optimal_lr:.2e}")

# Update LR
for pg in optimizer.param_groups:
    pg['lr'] = optimal_lr

# 5. Train
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.use_amp = True
trainer.clip_grad_norm = 1.0
trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = trainer.train(epochs=EPOCHS, early_stopping_patience=15)

print(f"✓ Best Val Loss: {min(history['val_loss']):.4f}")
```

### Template 2: Transfer Learning

```python
#%% python
"""
Transfer Learning Template
Modify: num_classes, freeze_until layer
"""

from model_builder import ModelTemplates, ModelSurgery

NUM_CLASSES = 5  # Your dataset classes

# 1. Load pretrained
model = ModelTemplates.resnet18(num_classes=1000)  # ImageNet weights

# 2. Freeze layers
ModelSurgery.freeze_layers(model, until='layer3')

# 3. Replace head
ModelSurgery.replace_head(model, num_classes=NUM_CLASSES)

# 4. Check what's trainable
params = ModelSurgery.count_parameters(model)
print(f"Trainable: {params['trainable']:,} / {params['total']:,}")

# 5. Train head only (few epochs)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3
)
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.train(epochs=5)

# 6. Unfreeze & fine-tune
ModelSurgery.unfreeze_layers(model, from_layer='layer3')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower LR
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.train(epochs=20)
```

### Template 3: Custom CNN

```python
#%% python
"""
Build Custom CNN from Scratch
"""

import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes=10)
print(model)

# Summary
from model_builder import ModelSummary
ModelSummary.summary(model, input_size=(3, 32, 32))
```

### Template 4: Debugging Session

```python
#%% python
"""
Complete Debugging Workflow
Run this if training fails or behaves strangely
"""

from debugging import *

# 1. Health Check
print("🏥 Running Health Check...")
report = ModelHealthCheck.check(model, sample_input)

# 2. NaN Detection
print("\n🔍 Setting up NaN Detector...")
detector = NaNDetector(model)
detector.register_hooks()

# 3. Gradient Check
print("\n✅ Checking Gradients...")
model.zero_grad()
output = model(sample_input)
loss = criterion(output, target)
loss.backward()

grad_stats = GradientChecker.check_gradients(model)

# 4. Memory Profile
print("\n💾 Profiling Memory...")
memory_stats = MemoryProfiler.profile(model, input_size=(3, 224, 224))

# 5. Speed Profile
print("\n⏱️ Profiling Speed...")
speed_stats = SpeedProfiler.profile(model, input_size=(3, 224, 224), num_runs=100)

# 6. Shape Trace
print("\n📐 Tracing Shapes...")
shapes = ShapeTracer.trace(model, input_size=(3, 224, 224))

# 7. Backward Debug
print("\n🔙 Debugging Backward Pass...")
debugger = BackwardDebugger(model)
debugger.register_hooks()

output = model(sample_input)
loss = criterion(output, target)
loss.backward()

debugger.print_stats()
debugger.remove_hooks()

detector.remove_hooks()
print("\n✓ Debugging Complete!")
```

---

## 🎯 Common Workflows

### Workflow 1: Starting a New Project

```python
# Step 1: Analyze your data
stats = DatasetAnalyzer.analyze(dataset)
corrupted = DatasetAnalyzer.find_corrupted(dataset)

# Step 2: Preview augmentations
previews = AugmentationPreview.preview(dataset, transform, num_samples=9)

# Step 3: Split data
train_ds, val_ds, test_ds = DataSplitter.stratified_split(dataset)

# Step 4: Create loaders
train_loader = SmartDataLoader.create(train_ds, batch_size=64)
val_loader = SmartDataLoader.create(val_ds, batch_size=64, shuffle=False)

# Step 5: Choose model
model = ModelTemplates.resnet18(num_classes=NUM_CLASSES)

# Step 6: Find LR
lr_finder = LRFinder(model, optimizer, criterion)
results = lr_finder.range_test(train_loader)
optimal_lr = results['optimal_lr']

# Step 7: Train
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.use_amp = True
history = trainer.train(epochs=100)
```

### Workflow 2: Improving Existing Model

```python
# Step 1: Profile current model
memory_stats = MemoryProfiler.profile(model, input_size=(3, 224, 224))
speed_stats = SpeedProfiler.profile(model, input_size=(3, 224, 224))

# Step 2: Add regularization
model = ModelSurgery.add_dropout(model, p=0.5)

# Step 3: Try different optimizers
optimizers = [
    torch.optim.Adam(model.parameters(), lr=1e-3),
    torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9),
    torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
]

# Train each and compare
results = {}
for opt in optimizers:
    trainer = AutoTrainer(model, train_loader, val_loader, opt, criterion)
    history = trainer.train(epochs=10)
    results[opt.__class__.__name__] = min(history['val_loss'])

print(results)
```

### Workflow 3: Deploying Model

```python
# Step 1: Load best checkpoint
checkpoint = torch.load('./checkpoints/best_model_epoch_45.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Step 2: Quantize
import torch.quantization as quantization
quantized_model = quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Step 3: Export ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Step 4: TorchScript
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')

print("✓ Model exported for deployment")
```

---

## 🔧 Configuration Tips

### GPU Optimization

```python
# Enable all optimizations
trainer.use_amp = True  # Mixed precision
trainer.gradient_accumulation_steps = 4  # Larger effective batch
trainer.clip_grad_norm = 1.0  # Stable training

# Optimal data loading
loader = SmartDataLoader.create(
    dataset,
    batch_size=256,  # As large as GPU can handle
    num_workers=8,
    pin_memory=True
)
```

### CPU Training

```python
# Optimize for CPU
trainer = AutoTrainer(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion,
    device='cpu'
)
trainer.use_amp = False  # AMP only for GPU
```

### Small GPU Memory

```python
# Use gradient accumulation
trainer.gradient_accumulation_steps = 8  # Effective batch *= 8

# Smaller batch size
loader = SmartDataLoader.create(dataset, batch_size=16)

# Mixed precision helps
trainer.use_amp = True
```

---

## 📊 Monitoring Training

### TensorBoard

```bash
# In terminal
tensorboard --logdir=./runs

# Open browser to http://localhost:6006
```

### Custom Logging

```python
# Access history during training
history = trainer.train(epochs=50)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 🆘 Troubleshooting

### Problem: Out of Memory

```python
# Solution 1: Reduce batch size
loader = SmartDataLoader.create(dataset, batch_size=16)

# Solution 2: Gradient accumulation
trainer.gradient_accumulation_steps = 8

# Solution 3: Profile memory
memory_stats = MemoryProfiler.profile(model, input_size=(3, 224, 224))
# Find memory-intensive layers
```

### Problem: NaN Loss

```python
# Solution 1: Enable NaN detector
detector = NaNDetector(model)
detector.register_hooks()

# Solution 2: Clip gradients
trainer.clip_grad_norm = 1.0

# Solution 3: Check data
stats = DatasetAnalyzer.analyze(dataset)
corrupted = DatasetAnalyzer.find_corrupted(dataset)

# Solution 4: Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Try smaller
```

### Problem: No Improvement

```python
# Check 1: Gradients flowing?
GradientChecker.check_gradients(model)

# Check 2: Learning rate too high/low?
lr_finder = LRFinder(model, optimizer, criterion)
results = lr_finder.range_test(train_loader)

# Check 3: Data augmentation working?
previews = AugmentationPreview.preview(dataset, transform)

# Check 4: Model capacity?
params = ModelSurgery.count_parameters(model)
print(f"Parameters: {params['total']:,}")
# Try bigger/smaller model
```

### Problem: Slow Training

```python
# Check 1: Profile speed
speed_stats = SpeedProfiler.profile(model, input_size=(3, 224, 224))

# Check 2: Enable mixed precision
trainer.use_amp = True

# Check 3: Optimize data loading
loader = SmartDataLoader.create(
    dataset,
    batch_size=128,  # Larger batches
    num_workers=8
)
```

---

## 🎓 Learning Path

### Beginner Level
1. Start with `Template 1: Image Classification`
2. Use `ModelTemplates` - don't build from scratch yet
3. Enable `AutoTrainer` features one by one
4. Visualize with TensorBoard

### Intermediate Level
1. Use `LRFinder` for every experiment
2. Try `Transfer Learning` (Template 2)
3. Profile your models (memory, speed)
4. Handle imbalanced datasets

### Advanced Level
1. Build custom architectures
2. Use all debugging tools
3. Implement custom training loops
4. Deploy with quantization/ONNX

---

**🔥 You're ready to build amazing deep learning models!**

For questions or issues, check the main README.md in `src/pytorch/`
