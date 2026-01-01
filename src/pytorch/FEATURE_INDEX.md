# 🎯 PyTorch IDE - Complete Feature Index

Quick reference for all 50+ features with copy-paste code snippets.

---

## 📑 Table of Contents

1. [Training (7 features)](#training-features)
2. [Architecture (6 features)](#architecture-features)
3. [Data Pipeline (10 features)](#data-pipeline-features)
4. [Debugging (8 features)](#debugging-features)
5. [Optimization (5 features)](#optimization-features)
6. [Checkpointing (4 features)](#checkpointing-features)
7. [Visualization (5+ features)](#visualization-features)
8. [Deployment (5 features)](#deployment-features)
9. [Utilities (10+ features)](#utilities-features)

---

## Training Features

### 1. Auto Training Loop
**File**: `trainer.py` | **Class**: `AutoTrainer`

```python
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
history = trainer.train(epochs=50, early_stopping_patience=10)
```

**Features**:
- Automatic epoch management
- Validation after each epoch
- Progress tracking
- Metric logging
- Checkpoint saving

---

### 2. Learning Rate Finder
**File**: `trainer.py` | **Class**: `LRFinder`

```python
lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
results = lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
optimal_lr = results['optimal_lr']
print(f"Use LR: {optimal_lr:.2e}")
```

**Output**: `{'lrs': [...], 'losses': [...], 'optimal_lr': 0.001}`

---

### 3. Mixed Precision Training
**File**: `trainer.py` | **Property**: `use_amp`

```python
trainer = AutoTrainer(model, train_loader, val_loader, optimizer, criterion)
trainer.use_amp = True  # Enable FP16 training
history = trainer.train(epochs=50)
```

**Speedup**: 2-3x faster on NVIDIA GPUs with Tensor Cores

---

### 4. Gradient Accumulation
**File**: `trainer.py` | **Property**: `gradient_accumulation_steps`

```python
trainer.gradient_accumulation_steps = 4
# Effective batch size = batch_size * 4
```

**Use case**: Train with larger effective batch sizes on limited GPU memory

---

### 5. Smart Early Stopping
**File**: `trainer.py` | **Class**: `EarlyStopping`

```python
# Built into AutoTrainer
history = trainer.train(epochs=100, early_stopping_patience=15)
# Stops if val_loss doesn't improve for 15 epochs
```

**Configuration**:
```python
early_stop = EarlyStopping(
    patience=10,
    min_delta=0.001,
    monitor='val_loss',
    mode='min'
)
```

---

### 6. Auto-Resume Training
**File**: `trainer.py` | **Property**: `auto_resume`

```python
trainer.auto_resume = True
# Automatically resumes from last checkpoint if exists
```

**No action needed** - crashes are handled automatically

---

### 7. Curriculum Learning
**File**: `trainer.py` | **Property**: `curriculum_fn`

```python
# Gradually increase difficulty
trainer.curriculum_fn = lambda epoch: min(1.0, epoch/10)

# Or custom function
def curriculum_schedule(epoch):
    if epoch < 10:
        return 0.3  # Easy samples
    elif epoch < 20:
        return 0.6  # Medium
    else:
        return 1.0  # All samples
        
trainer.curriculum_fn = curriculum_schedule
```

---

## Architecture Features

### 8. Model Templates - ResNet
**File**: `model_builder.py` | **Class**: `ModelTemplates`

```python
model = ModelTemplates.resnet18(num_classes=10, input_channels=3)
```

**Available**: ResNet18

---

### 9. Model Templates - U-Net
**File**: `model_builder.py`

```python
model = ModelTemplates.unet(in_channels=3, out_channels=1)
```

**Use case**: Segmentation tasks

---

### 10. Model Templates - Transformer
**File**: `model_builder.py`

```python
model = ModelTemplates.transformer_encoder(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048
)
```

---

### 11. Model Templates - LSTM
**File**: `model_builder.py`

```python
model = ModelTemplates.lstm_classifier(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    num_classes=10
)
```

---

### 12. Model Templates - VAE & GAN
**File**: `model_builder.py`

```python
# VAE
vae = ModelTemplates.vae(input_dim=784, latent_dim=20)

# GAN
gan = ModelTemplates.gan(latent_dim=100, img_channels=1, img_size=28)
generator = gan['generator']
discriminator = gan['discriminator']
```

---

### 13. Model Surgery - Freeze Layers
**File**: `model_builder.py` | **Class**: `ModelSurgery`

```python
# Freeze all layers up to layer3
ModelSurgery.freeze_layers(model, until='layer3')

# Freeze all layers
ModelSurgery.freeze_layers(model, until=None)

# Unfreeze from layer4 onwards
ModelSurgery.unfreeze_layers(model, from_layer='layer4')
```

---

### 14. Model Surgery - Replace Head
**File**: `model_builder.py`

```python
# Replace final classification layer
ModelSurgery.replace_head(model, num_classes=100, layer_name='fc')
```

---

### 15. Model Surgery - Add Dropout
**File**: `model_builder.py`

```python
model = ModelSurgery.add_dropout(model, p=0.5)
# Adds dropout after every ReLU activation
```

---

### 16. Model Summary
**File**: `model_builder.py` | **Class**: `ModelSummary`

```python
ModelSummary.summary(model, input_size=(3, 224, 224), device='cpu')
```

**Output**:
```
==================================================
Layer (type)               Output Shape         Param #
==================================================
Conv2d-1                   [1, 64, 112, 112]    9,408
BatchNorm2d-2              [1, 64, 112, 112]    128
ReLU-3                     [1, 64, 112, 112]    0
...
==================================================
Total params: 11,689,512
Trainable params: 11,689,512
Model size: 44.59 MB
==================================================
```

---

### 17. Parameter Counter
**File**: `model_builder.py`

```python
params = ModelSurgery.count_parameters(model)
print(f"Total: {params['total']:,}")
print(f"Trainable: {params['trainable']:,}")
print(f"Frozen: {params['frozen']:,}")
```

---

## Data Pipeline Features

### 18. Smart DataLoader
**File**: `data_pipeline.py` | **Class**: `SmartDataLoader`

```python
loader = SmartDataLoader.create(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=None,  # Auto-determined
    pin_memory=None     # Auto-determined based on CUDA
)
```

**Auto-optimizes**:
- `num_workers`: Based on CPU count
- `pin_memory`: Based on CUDA availability
- `persistent_workers`: Enabled if num_workers > 0

---

### 19-21. Augmentation Pipelines
**File**: `data_pipeline.py` | **Class**: `AugmentationPipeline`

```python
# Standard training augmentation
transform_train = AugmentationPipeline.image_classification_train(img_size=224)

# Validation (minimal augmentation)
transform_val = AugmentationPipeline.image_classification_val(img_size=224)

# Strong augmentation for difficult datasets
transform_strong = AugmentationPipeline.strong_augmentation(img_size=224)

# Cutout augmentation
cutout = AugmentationPipeline.cutout(n_holes=1, length=16)

# MixUp (during training)
mixed_x, y_a, y_b, lam = AugmentationPipeline.mixup_batch(x, y, alpha=1.0)
loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

### 22. Dataset Analyzer
**File**: `data_pipeline.py` | **Class**: `DatasetAnalyzer`

```python
stats = DatasetAnalyzer.analyze(dataset, num_samples=1000)
```

**Returns**:
```python
{
    'size': 50000,
    'num_classes': 10,
    'class_distribution': {0: 5000, 1: 5000, ...},
    'is_balanced': True,
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2470, 0.2435, 0.2616]
}
```

---

### 23. Find Corrupted Samples
**File**: `data_pipeline.py`

```python
corrupted_indices = DatasetAnalyzer.find_corrupted(dataset)
print(f"Found {len(corrupted_indices)} corrupted samples")

# Remove from dataset
clean_indices = [i for i in range(len(dataset)) if i not in corrupted_indices]
clean_dataset = Subset(dataset, clean_indices)
```

---

### 24. Imbalanced Data Handler
**File**: `data_pipeline.py` | **Class**: `ImbalancedDataHandler`

```python
# Option 1: Weighted sampler
sampler = ImbalancedDataHandler.create_weighted_sampler(dataset)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)

# Option 2: Balanced loader (all-in-one)
loader = ImbalancedDataHandler.create_balanced_loader(dataset, batch_size=64)
```

---

### 25. Augmentation Preview
**File**: `data_pipeline.py` | **Class**: `AugmentationPreview`

```python
previews = AugmentationPreview.preview(
    dataset,
    transform,
    num_samples=9,
    num_augmentations=4
)

# Returns list of dicts:
# [
#   {
#     'original': <image>,
#     'augmented': [<aug1>, <aug2>, <aug3>, <aug4>],
#     'label': 5
#   },
#   ...
# ]
```

---

### 26-27. Data Splitting
**File**: `data_pipeline.py` | **Class**: `DataSplitter`

```python
# Random split
train_ds, val_ds, test_ds = DataSplitter.split(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)

# Stratified split (preserves class distribution)
train_ds, val_ds, test_ds = DataSplitter.stratified_split(
    dataset,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_seed=42
)
```

---

## Debugging Features

### 28. NaN/Inf Detector
**File**: `debugging.py` | **Class**: `NaNDetector`

```python
detector = NaNDetector(model)
detector.register_hooks()

# Training loop
for epoch in range(epochs):
    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if detector.nan_found:
            print(f"NaN detected in: {detector.nan_location}")
            break

detector.remove_hooks()
```

**Auto-stops** on first NaN/Inf and reports location

---

### 29. Gradient Checker
**File**: `debugging.py` | **Class**: `GradientChecker`

```python
# After backward pass
loss.backward()

stats = GradientChecker.check_gradients(model, verbose=True)

# Returns:
# {
#   'layer1.weight': {
#     'has_gradient': True,
#     'norm': 0.1234,
#     'mean': 0.0001,
#     'max': 0.5,
#     'min': -0.3
#   },
#   ...
# }
```

---

### 30. Gradient Flow Plot
**File**: `debugging.py`

```python
flow_data = GradientChecker.plot_gradient_flow(model)

# Returns data for plotting:
# {
#   'layers': ['conv1', 'bn1', 'relu', ...],
#   'average_gradients': [0.1, 0.2, 0.15, ...],
#   'max_gradients': [0.5, 0.8, 0.6, ...]
# }

# Plot with matplotlib:
import matplotlib.pyplot as plt
plt.plot(flow_data['average_gradients'])
plt.xlabel('Layer')
plt.ylabel('Average Gradient')
plt.show()
```

---

### 31. Memory Profiler
**File**: `debugging.py` | **Class**: `MemoryProfiler`

```python
memory_stats = MemoryProfiler.profile(
    model,
    input_size=(3, 224, 224),
    device='cuda'
)

# Output:
# Layer                          Allocated (MB)    Reserved (MB)
# Conv2d                                  45.23            48.00
# BatchNorm2d                             45.67            48.00
# ...
# Peak memory usage: 1234.56 MB
```

---

### 32. Speed Profiler
**File**: `debugging.py` | **Class**: `SpeedProfiler`

```python
speed_stats = SpeedProfiler.profile(
    model,
    input_size=(3, 224, 224),
    device='cuda',
    num_runs=100
)

# Returns:
# {
#   'total_time_ms': 15.23,
#   'layer_times': {
#     'conv1': 2.34,
#     'bn1': 0.45,
#     ...
#   }
# }
```

---

### 33. Shape Tracer
**File**: `debugging.py` | **Class**: `ShapeTracer`

```python
shapes = ShapeTracer.trace(model, input_size=(3, 224, 224), device='cpu')

# Output:
# Layer                     Input Shape         Output Shape
# conv1                     [1, 3, 224, 224]    [1, 64, 112, 112]
# bn1                       [1, 64, 112, 112]   [1, 64, 112, 112]
# ...
```

---

### 34. Backward Debugger
**File**: `debugging.py` | **Class**: `BackwardDebugger`

```python
debugger = BackwardDebugger(model)
debugger.register_hooks()

# Training loop
output = model(data)
loss = criterion(output, target)
loss.backward()

debugger.print_stats()
debugger.remove_hooks()

# Output:
# Layer              Mean        Std        Min        Max        Norm
# conv1.weight     0.0001     0.0234    -0.1234     0.2345     1.2345
# ...
```

---

### 35. Model Health Check
**File**: `debugging.py` | **Class**: `ModelHealthCheck`

```python
report = ModelHealthCheck.check(model, sample_input)

# Returns:
# {
#   'parameters': {'total': 11689512, 'trainable': 11689512, 'frozen': 0},
#   'forward_pass': 'OK',
#   'output_validity': 'OK',
#   'gradients': 'OK'
# }
```

---

## Optimization Features

### 36. Learning Rate Schedulers
**File**: `trainer.py` | **Property**: `scheduler`

```python
from torch.optim.lr_scheduler import *

# Cosine Annealing (recommended)
trainer.scheduler = CosineAnnealingLR(optimizer, T_max=100)

# OneCycleLR (very effective)
trainer.scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader)
)

# Reduce on plateau
trainer.scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)

# Step decay
trainer.scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

---

### 37. Gradient Clipping
**File**: `trainer.py` | **Property**: `clip_grad_norm`

```python
trainer.clip_grad_norm = 1.0  # Clip at norm=1.0
# Essential for RNNs and Transformers
```

---

### 38. Weight Decay
**Integrated with optimizers**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

---

### 39. Multiple Optimizer Comparison

```python
results = {}

for opt_class, lr in [(Adam, 1e-3), (SGD, 1e-2), (AdamW, 1e-3)]:
    model_copy = copy.deepcopy(model)
    optimizer = opt_class(model_copy.parameters(), lr=lr)
    trainer = AutoTrainer(model_copy, train_loader, val_loader, optimizer, criterion)
    history = trainer.train(epochs=20)
    results[opt_class.__name__] = min(history['val_loss'])

print(f"Best optimizer: {min(results, key=results.get)}")
```

---

### 40. Automatic Batch Size Finder

```python
# Find maximum batch size that fits in memory
def find_max_batch_size(model, dataset, start_size=16, max_size=512):
    for batch_size in [16, 32, 64, 128, 256, 512]:
        if batch_size > max_size:
            break
        try:
            loader = DataLoader(dataset, batch_size=batch_size)
            batch = next(iter(loader))
            _ = model(batch[0].cuda())
            print(f"✓ Batch size {batch_size} works")
        except RuntimeError:
            print(f"✗ Batch size {batch_size} OOM")
            return batch_size // 2
    return max_size

optimal_batch = find_max_batch_size(model, dataset)
```

---

## Checkpointing Features

### 41. Checkpoint Manager
**File**: `trainer.py` | **Class**: `CheckpointManager`

```python
ckpt_manager = CheckpointManager(save_dir='./checkpoints', keep_best=3, keep_last=2)

# Save
ckpt_manager.save(model, optimizer, epoch=10, metrics={'val_loss': 0.5}, is_best=True)

# Load best
best_checkpoint = ckpt_manager.load_best()
model.load_state_dict(best_checkpoint['model_state_dict'])

# Load last (for resuming)
last_checkpoint = ckpt_manager.load_last()
```

---

### 42. Best Model Tracking
**Built into AutoTrainer and CheckpointManager**

```python
# Automatically saves top-N models by validation loss
# Access via:
best_checkpoint = trainer.checkpoint_manager.load_best()
```

---

### 43. TensorBoard Integration
**Built into AutoTrainer**

```python
trainer = AutoTrainer(
    model, train_loader, val_loader, optimizer, criterion,
    log_dir='./runs/experiment1'
)

# Logs automatically:
# - Training/validation loss
# - Training/validation accuracy
# - Learning rate
# - Gradients
# - Weights

# View in terminal:
# tensorboard --logdir=./runs
```

---

### 44. Training History Export
**File**: `trainer.py`

```python
history = trainer.train(epochs=50)

# Auto-saved to: checkpoints/history.json
# {
#   'train_loss': [2.3, 1.9, 1.5, ...],
#   'val_loss': [2.1, 1.8, 1.4, ...],
#   'train_acc': [20, 35, 50, ...],
#   'val_acc': [25, 40, 55, ...],
#   'lr': [0.001, 0.001, 0.0009, ...]
# }

# Load and plot
import json
with open('checkpoints/history.json') as f:
    history = json.load(f)
```

---

## Visualization Features

### 45-49. Training Visualizations

```python
import matplotlib.pyplot as plt

# Loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

# Learning rate schedule
plt.plot(history['lr'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

preds, labels = [], []
for x, y in val_loader:
    pred = model(x.cuda()).argmax(1).cpu()
    preds.extend(pred.numpy())
    labels.extend(y.numpy())

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

# Feature visualization
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer4.register_forward_hook(get_activation('layer4'))
_ = model(image)
features = activations['layer4']

# Visualize feature maps
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    if i < features.size(1):
        ax.imshow(features[0, i].cpu(), cmap='viridis')
    ax.axis('off')
plt.show()
```

---

## Deployment Features

### 50. ONNX Export

```python
model.eval()
dummy_input = torch.randn(1, 3, 224, 224).cuda()

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✓ Exported to ONNX")

# Verify
import onnx
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
```

---

### 51. TorchScript Conversion

```python
model.eval()

# Tracing
example_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, example_input)
traced.save('model_traced.pt')

# Scripting (for control flow)
scripted = torch.jit.script(model)
scripted.save('model_scripted.pt')

# Load
loaded = torch.jit.load('model_scripted.pt')
```

---

### 52. Quantization

```python
import torch.quantization as quantization

model.eval()

# Dynamic quantization (easiest)
quantized = quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# Static quantization (more work, better results)
model.qconfig = quantization.get_default_qconfig('fbgemm')
quantization.prepare(model, inplace=True)

# Calibrate with representative data
with torch.no_grad():
    for data, _ in calibration_loader:
        model(data)

quantized = quantization.convert(model, inplace=False)

# Save
torch.save(quantized.state_dict(), 'model_quantized.pth')

# Size comparison
original_size = os.path.getsize('model.pth') / (1024**2)
quantized_size = os.path.getsize('model_quantized.pth') / (1024**2)
print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB ({original_size/quantized_size:.1f}x smaller)")
```

---

### 53. Model Pruning

```python
import torch.nn.utils.prune as prune

# Prune individual layer
prune.l1_unstructured(model.conv1, name='weight', amount=0.3)

# Prune multiple layers
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3
)

# Make pruning permanent
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# Check sparsity
def check_sparsity(model):
    zeros = 0
    total = 0
    for param in model.parameters():
        zeros += (param == 0).sum().item()
        total += param.numel()
    return 100. * zeros / total

print(f"Sparsity: {check_sparsity(model):.2f}%")
```

---

### 54. Inference Optimization

```python
# 1. Fuse layers
model.eval()
model = torch.quantization.fuse_modules(
    model,
    [['conv1', 'bn1', 'relu']],
    inplace=True
)

# 2. Optimize for inference
model = torch.jit.optimize_for_inference(torch.jit.script(model))

# 3. Use inference mode (PyTorch 1.9+)
with torch.inference_mode():
    output = model(input)
```

---

## Utilities Features

### 55. Reproducibility Lock

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

### 56. Environment Save

```python
import subprocess

# Save requirements
subprocess.run(['pip', 'freeze', '>', 'requirements.txt'], shell=True)

# Save system info
info = {
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
}

import json
with open('environment.json', 'w') as f:
    json.dump(info, f, indent=2)
```

---

### 57-60. Experiment Tracking

```python
# Built into AutoTrainer
# Automatically tracks:
# - Hyperparameters
# - Metrics (loss, accuracy)
# - Checkpoints
# - Training history
# - TensorBoard logs

# Access:
history = trainer.train(epochs=50)
best_checkpoint = trainer.checkpoint_manager.load_best()
```

---

## 🔍 Quick Search

Looking for something specific?

- **Speed up training** → Features #3, #4, #36
- **Fix NaN loss** → Features #28, #37
- **Out of memory** → Features #4, #31, #40
- **No improvement** → Features #2, #29, #39
- **Imbalanced dataset** → Feature #24
- **Transfer learning** → Features #13, #14
- **Deploy model** → Features #50, #51, #52
- **Debug gradients** → Features #29, #30, #34
- **Profile model** → Features #31, #32, #33

---

**🎯 Index Complete! All 50+ features documented with examples.**
