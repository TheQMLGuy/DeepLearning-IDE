"""
PyTorch Debugging Tools - NaN detection, gradient checking, profiling
Features: Memory profiler, speed profiler, gradient checker, shape tracer
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import time
from collections import defaultdict


class NaNDetector:
    """Detect NaN/Inf values during training."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.nan_found = False
        self.nan_location = None
        
    def register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    self.nan_found = True
                    self.nan_location = f"Forward: {module.__class__.__name__}"
                    print(f"⚠️  NaN/Inf detected in {self.nan_location}")
                    print(f"   Output stats: min={output.min():.4f}, max={output.max():.4f}")
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                if torch.isnan(grad_output[0]).any() or torch.isinf(grad_output[0]).any():
                    self.nan_found = True
                    self.nan_location = f"Backward: {module.__class__.__name__}"
                    print(f"⚠️  NaN/Inf detected in {self.nan_location}")
        
        for module in self.model.modules():
            self.hooks.append(module.register_forward_hook(forward_hook))
            self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GradientChecker:
    """Check gradient flow and detect dead neurons."""
    
    @staticmethod
    def check_gradients(model: nn.Module, verbose: bool = True) -> Dict:
        """Check which layers have gradients flowing."""
        stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                
                has_gradient = grad_norm > 1e-8
                
                stats[name] = {
                    'has_gradient': has_gradient,
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'max': grad_max,
                    'min': grad_min
                }
                
                if verbose:
                    status = "✓" if has_gradient else "✗"
                    print(f"{status} {name:50s} | norm: {grad_norm:.6f}")
        
        # Summary
        total = len(stats)
        with_grad = sum(1 for s in stats.values() if s['has_gradient'])
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Gradient Flow: {with_grad}/{total} layers have gradients")
            if with_grad < total:
                print(f"⚠️  {total - with_grad} layers have zero gradients (dead neurons?)")
            print(f"{'='*70}\n")
        
        return stats
    
    @staticmethod
    def plot_gradient_flow(model: nn.Module):
        """Generate data for plotting gradient flow."""
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        return {
            'layers': layers,
            'average_gradients': ave_grads,
            'max_gradients': max_grads
        }


class MemoryProfiler:
    """Profile GPU memory usage."""
    
    @staticmethod
    def profile(model: nn.Module, input_size: tuple, device: str = 'cuda'):
        """Profile memory usage per layer."""
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, skipping memory profiling")
            return None
        
        model = model.to(device)
        model.train()
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create input
        x = torch.randn(1, *input_size).to(device)
        
        memory_stats = []
        
        def hook_fn(module, input, output):
            torch.cuda.synchronize()
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
            
            memory_stats.append({
                'layer': module.__class__.__name__,
                'allocated_mb': memory_used,
                'reserved_mb': memory_reserved
            })
        
        # Register hooks
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Print summary
        print(f"\n{'='*70}")
        print("Memory Profile")
        print(f"{'='*70}")
        print(f"{'Layer':<30} {'Allocated (MB)':>15} {'Reserved (MB)':>15}")
        print(f"{'-'*70}")
        
        for stat in memory_stats[-10:]:  # Show last 10 layers
            print(f"{stat['layer']:<30} {stat['allocated_mb']:>15.2f} {stat['reserved_mb']:>15.2f}")
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{'-'*70}")
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        print(f"{'='*70}\n")
        
        return memory_stats


class SpeedProfiler:
    """Profile inference speed per layer."""
    
    @staticmethod
    def profile(model: nn.Module, input_size: tuple, device: str = 'cuda', num_runs: int = 100):
        """Profile inference speed."""
        model = model.to(device).eval()
        x = torch.randn(1, *input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Profile layers
        layer_times = defaultdict(list)
        
        def hook_fn(name):
            def hook(module, input, output):
                if device == 'cuda':
                    torch.cuda.synchronize()
                layer_times[name].append(time.time())
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Time multiple runs
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        total_time = (time.time() - start_time) / num_runs * 1000  # ms
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate per-layer times
        layer_stats = {}
        for name, times in layer_times.items():
            if len(times) >= 2:
                # Calculate average time between consecutive calls
                diffs = [(times[i+1] - times[i]) * 1000 for i in range(0, len(times)-1, 2)]
                avg_time = sum(diffs) / len(diffs) if diffs else 0
                layer_stats[name] = avg_time
        
        # Print summary
        print(f"\n{'='*70}")
        print("Speed Profile")
        print(f"{'='*70}")
        print(f"Total inference time: {total_time:.2f} ms")
        print(f"\n{'Layer':<50} {'Time (ms)':>15}")
        print(f"{'-'*70}")
        
        # Sort by time
        sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1], reverse=True)
        for name, layer_time in sorted_layers[:15]:  # Show top 15
            percentage = 100 * layer_time / total_time
            print(f"{name:<50} {layer_time:>10.3f} ({percentage:>4.1f}%)")
        
        print(f"{'='*70}\n")
        
        return {
            'total_time_ms': total_time,
            'layer_times': layer_stats
        }


class ShapeTracer:
    """Trace tensor shapes through the network."""
    
    @staticmethod
    def trace(model: nn.Module, input_size: tuple, device: str = 'cpu'):
        """Trace shapes through network."""
        model = model.to(device)
        
        shapes = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    shapes.append({
                        'layer': name,
                        'input_shape': list(input[0].shape) if isinstance(input, tuple) else list(input.shape),
                        'output_shape': list(output.shape)
                    })
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        x = torch.randn(1, *input_size).to(device)
        with torch.no_grad():
            _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Print
        print(f"\n{'='*80}")
        print("Shape Trace")
        print(f"{'='*80}")
        print(f"{'Layer':<40} {'Input Shape':<20} {'Output Shape':<20}")
        print(f"{'-'*80}")
        
        for shape_info in shapes:
            print(f"{shape_info['layer']:<40} "
                  f"{str(shape_info['input_shape']):<20} "
                  f"{str(shape_info['output_shape']):<20}")
        
        print(f"{'='*80}\n")
        
        return shapes


class BackwardDebugger:
    """Debug backward pass and gradient flow."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_stats = {}
        self.hooks = []
    
    def register_hooks(self):
        """Register backward hooks."""
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad = grad_output[0]
                    self.gradient_stats[name] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'norm': grad.norm().item()
                    }
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                self.hooks.append(module.register_full_backward_hook(backward_hook(name)))
    
    def remove_hooks(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def print_stats(self):
        """Print gradient statistics."""
        print(f"\n{'='*90}")
        print("Gradient Statistics")
        print(f"{'='*90}")
        print(f"{'Layer':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Norm':>10}")
        print(f"{'-'*90}")
        
        for name, stats in self.gradient_stats.items():
            print(f"{name:<40} "
                  f"{stats['mean']:>10.6f} "
                  f"{stats['std']:>10.6f} "
                  f"{stats['min']:>10.6f} "
                  f"{stats['max']:>10.6f} "
                  f"{stats['norm']:>10.6f}")
        
        print(f"{'='*90}\n")


class ModelHealthCheck:
    """Comprehensive model health check."""
    
    @staticmethod
    def check(model: nn.Module, sample_input: torch.Tensor) -> Dict:
        """Run comprehensive health check."""
        print("🏥 Running Model Health Check...\n")
        
        report = {}
        
        # 1. Parameter check
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        report['parameters'] = {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
        
        print(f"✓ Parameters: {total_params:,} total ({trainable_params:,} trainable)")
        
        # 2. Forward pass
        try:
            with torch.no_grad():
                output = model(sample_input)
            report['forward_pass'] = 'OK'
            print(f"✓ Forward pass: OK (output shape: {output.shape})")
        except Exception as e:
            report['forward_pass'] = f'FAILED: {str(e)}'
            print(f"✗ Forward pass: FAILED - {e}")
            return report
        
        # 3. NaN/Inf check
        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()
        
        if has_nan or has_inf:
            report['output_validity'] = 'FAILED'
            print(f"✗ Output validity: NaN={has_nan}, Inf={has_inf}")
        else:
            report['output_validity'] = 'OK'
            print(f"✓ Output validity: No NaN/Inf")
        
        # 4. Gradient check
        model.zero_grad()
        loss = output.sum()
        loss.backward()
        
        has_gradients = all(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.parameters() if p.requires_grad
        )
        
        if has_gradients:
            report['gradients'] = 'OK'
            print(f"✓ Gradients: Flowing correctly")
        else:
            report['gradients'] = 'WARNING'
            print(f"⚠️  Gradients: Some layers have zero gradients")
        
        print(f"\n{'='*60}")
        print("Health Check Complete")
        print(f"{'='*60}\n")
        
        return report


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Dummy input
    x = torch.randn(4, 784)
    
    # NaN Detection
    detector = NaNDetector(model)
    detector.register_hooks()
    
    # Gradient checking
    y = model(x)
    loss = y.sum()
    loss.backward()
    GradientChecker.check_gradients(model)
    
    # Shape tracing
    ShapeTracer.trace(model, input_size=(784,))
    
    # Health check
    ModelHealthCheck.check(model, x)
    
    detector.remove_hooks()
