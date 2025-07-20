"""
SpaceTimeCore: Shared foundation for all space-time optimization tools

This module provides the core functionality that all tools build upon:
- Memory profiling and hierarchy modeling
- √n interval calculation based on Williams' bound
- Strategy comparison framework
- Resource-aware scheduling
"""

import numpy as np
import psutil
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional
from enum import Enum
import json
import matplotlib.pyplot as plt


class OptimizationStrategy(Enum):
    """Different space-time tradeoff strategies"""
    CONSTANT = "constant"      # O(1) space
    LOGARITHMIC = "logarithmic"  # O(log n) space
    SQRT_N = "sqrt_n"           # O(√n) space - Williams' bound
    LINEAR = "linear"           # O(n) space
    ADAPTIVE = "adaptive"       # Dynamically chosen


@dataclass
class MemoryHierarchy:
    """Model of system memory hierarchy"""
    l1_size: int          # L1 cache size in bytes
    l2_size: int          # L2 cache size in bytes
    l3_size: int          # L3 cache size in bytes
    ram_size: int         # RAM size in bytes
    disk_size: int        # Available disk space in bytes
    
    l1_latency: float     # L1 access time in nanoseconds
    l2_latency: float     # L2 access time in nanoseconds
    l3_latency: float     # L3 access time in nanoseconds
    ram_latency: float    # RAM access time in nanoseconds
    disk_latency: float   # Disk access time in nanoseconds
    
    @classmethod
    def detect_system(cls) -> 'MemoryHierarchy':
        """Auto-detect system memory hierarchy"""
        # Default values for typical modern systems
        # In production, would use platform-specific detection
        return cls(
            l1_size=64 * 1024,        # 64KB
            l2_size=256 * 1024,       # 256KB
            l3_size=8 * 1024 * 1024,  # 8MB
            ram_size=psutil.virtual_memory().total,
            disk_size=psutil.disk_usage('/').free,
            l1_latency=1,             # 1ns
            l2_latency=4,             # 4ns
            l3_latency=12,            # 12ns
            ram_latency=100,          # 100ns
            disk_latency=10_000_000   # 10ms
        )
    
    def get_level_for_size(self, size_bytes: int) -> Tuple[str, float]:
        """Determine which memory level can hold the given size"""
        if size_bytes <= self.l1_size:
            return "L1", self.l1_latency
        elif size_bytes <= self.l2_size:
            return "L2", self.l2_latency
        elif size_bytes <= self.l3_size:
            return "L3", self.l3_latency
        elif size_bytes <= self.ram_size:
            return "RAM", self.ram_latency
        else:
            return "Disk", self.disk_latency


class SqrtNCalculator:
    """Calculate optimal √n intervals based on Williams' bound"""
    
    @staticmethod
    def calculate_interval(n: int, element_size: int = 8) -> int:
        """
        Calculate optimal checkpoint/buffer interval
        
        Args:
            n: Total number of elements
            element_size: Size of each element in bytes
            
        Returns:
            Optimal interval following √n pattern
        """
        # Basic √n calculation
        sqrt_n = int(np.sqrt(n))
        
        # Adjust for cache line alignment (typically 64 bytes)
        cache_line_size = 64
        elements_per_cache_line = cache_line_size // element_size
        
        # Round to nearest cache line boundary
        if sqrt_n > elements_per_cache_line:
            sqrt_n = (sqrt_n // elements_per_cache_line) * elements_per_cache_line
        
        return max(1, sqrt_n)
    
    @staticmethod
    def calculate_memory_usage(n: int, strategy: OptimizationStrategy, 
                             element_size: int = 8) -> int:
        """Calculate memory usage for different strategies"""
        if strategy == OptimizationStrategy.CONSTANT:
            return element_size * 10  # Small constant
        elif strategy == OptimizationStrategy.LOGARITHMIC:
            return element_size * int(np.log2(n) + 1)
        elif strategy == OptimizationStrategy.SQRT_N:
            return element_size * SqrtNCalculator.calculate_interval(n, element_size)
        elif strategy == OptimizationStrategy.LINEAR:
            return element_size * n
        else:  # ADAPTIVE
            # Choose based on available memory
            hierarchy = MemoryHierarchy.detect_system()
            if n * element_size <= hierarchy.l3_size:
                return element_size * n  # Fit in cache
            else:
                return element_size * SqrtNCalculator.calculate_interval(n, element_size)


class MemoryProfiler:
    """Profile memory usage patterns of functions"""
    
    def __init__(self):
        self.samples = []
        self.hierarchy = MemoryHierarchy.detect_system()
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict:
        """Profile a function's memory usage"""
        import tracemalloc
        
        # Start tracing
        tracemalloc.start()
        start_time = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        end_time = time.time()
        tracemalloc.stop()
        
        # Analyze memory level
        level, latency = self.hierarchy.get_level_for_size(peak)
        
        return {
            'result': result,
            'peak_memory': peak,
            'current_memory': current,
            'execution_time': end_time - start_time,
            'memory_level': level,
            'expected_latency': latency,
            'timestamp': time.time()
        }
    
    def compare_strategies(self, func: Callable, n: int, 
                          strategies: List[OptimizationStrategy]) -> Dict:
        """Compare different optimization strategies"""
        results = {}
        
        for strategy in strategies:
            # Configure function with strategy
            configured_func = lambda: func(n, strategy)
            
            # Profile it
            profile = self.profile_function(configured_func)
            results[strategy.value] = profile
        
        return results


class ResourceAwareScheduler:
    """Schedule operations based on available resources"""
    
    def __init__(self, memory_limit: Optional[int] = None):
        self.memory_limit = memory_limit or psutil.virtual_memory().available
        self.hierarchy = MemoryHierarchy.detect_system()
    
    def schedule_checkpoints(self, total_size: int, element_size: int = 8) -> List[int]:
        """
        Schedule checkpoint locations based on memory constraints
        
        Returns list of indices where checkpoints should occur
        """
        n = total_size // element_size
        
        # Calculate √n interval
        sqrt_interval = SqrtNCalculator.calculate_interval(n, element_size)
        
        # Adjust based on available memory
        if sqrt_interval * element_size > self.memory_limit:
            # Need smaller intervals
            adjusted_interval = self.memory_limit // element_size
        else:
            adjusted_interval = sqrt_interval
        
        # Generate checkpoint indices
        checkpoints = []
        for i in range(adjusted_interval, n, adjusted_interval):
            checkpoints.append(i)
        
        return checkpoints


class StrategyAnalyzer:
    """Analyze and visualize impact of different strategies"""
    
    @staticmethod
    def simulate_strategies(n_values: List[int], 
                          element_size: int = 8) -> Dict[str, Dict]:
        """Simulate different strategies across input sizes"""
        strategies = [
            OptimizationStrategy.CONSTANT,
            OptimizationStrategy.LOGARITHMIC,
            OptimizationStrategy.SQRT_N,
            OptimizationStrategy.LINEAR
        ]
        
        results = {strategy.value: {'n': [], 'memory': [], 'time': []} 
                  for strategy in strategies}
        
        hierarchy = MemoryHierarchy.detect_system()
        
        for n in n_values:
            for strategy in strategies:
                memory = SqrtNCalculator.calculate_memory_usage(n, strategy, element_size)
                
                # Simulate time based on memory level
                level, latency = hierarchy.get_level_for_size(memory)
                
                # Simple model: time = n * latency * recomputation_factor
                if strategy == OptimizationStrategy.CONSTANT:
                    time_estimate = n * latency * n  # O(n²) recomputation
                elif strategy == OptimizationStrategy.LOGARITHMIC:
                    time_estimate = n * latency * np.log2(n)
                elif strategy == OptimizationStrategy.SQRT_N:
                    time_estimate = n * latency * np.sqrt(n)
                else:  # LINEAR
                    time_estimate = n * latency
                
                results[strategy.value]['n'].append(n)
                results[strategy.value]['memory'].append(memory)
                results[strategy.value]['time'].append(time_estimate)
        
        return results
    
    @staticmethod
    def visualize_tradeoffs(results: Dict[str, Dict], save_path: str = None):
        """Create visualization comparing strategies"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot memory usage
        for strategy, data in results.items():
            ax1.loglog(data['n'], data['memory'], 'o-', label=strategy, linewidth=2)
        
        ax1.set_xlabel('Input Size (n)', fontsize=12)
        ax1.set_ylabel('Memory Usage (bytes)', fontsize=12)
        ax1.set_title('Memory Usage by Strategy', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot time complexity
        for strategy, data in results.items():
            ax2.loglog(data['n'], data['time'], 's-', label=strategy, linewidth=2)
        
        ax2.set_xlabel('Input Size (n)', fontsize=12)
        ax2.set_ylabel('Estimated Time (ns)', fontsize=12)
        ax2.set_title('Time Complexity by Strategy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Space-Time Tradeoffs: Strategy Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_recommendation(results: Dict[str, Dict], n: int) -> str:
        """Generate AI-style explanation of results"""
        # Find √n results
        sqrt_results = None
        linear_results = None
        
        for strategy, data in results.items():
            if strategy == OptimizationStrategy.SQRT_N.value:
                idx = data['n'].index(n) if n in data['n'] else -1
                if idx >= 0:
                    sqrt_results = {
                        'memory': data['memory'][idx],
                        'time': data['time'][idx]
                    }
            elif strategy == OptimizationStrategy.LINEAR.value:
                idx = data['n'].index(n) if n in data['n'] else -1
                if idx >= 0:
                    linear_results = {
                        'memory': data['memory'][idx],
                        'time': data['time'][idx]
                    }
        
        if sqrt_results and linear_results:
            memory_savings = (1 - sqrt_results['memory'] / linear_results['memory']) * 100
            time_increase = (sqrt_results['time'] / linear_results['time'] - 1) * 100
            
            return (
                f"√n checkpointing saved {memory_savings:.1f}% memory "
                f"with only {time_increase:.1f}% slowdown. "
                f"This function was recommended for checkpointing because "
                f"its memory growth exceeds √n relative to time."
            )
        
        return "Unable to generate recommendation - insufficient data"


# Export main components
__all__ = [
    'OptimizationStrategy',
    'MemoryHierarchy',
    'SqrtNCalculator',
    'MemoryProfiler',
    'ResourceAwareScheduler',
    'StrategyAnalyzer'
]