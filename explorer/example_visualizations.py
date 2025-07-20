#!/usr/bin/env python3
"""
Example visualizations demonstrating SpaceTime Explorer capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetime_explorer import SpaceTimeVisualizer
import matplotlib.pyplot as plt
import numpy as np


def visualize_algorithm_comparison():
    """Compare different algorithms visually"""
    print("="*60)
    print("Algorithm Comparison Visualization")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Space-Time Tradeoffs: Algorithm Comparison', fontsize=16)
    
    # Data range
    n_values = np.logspace(2, 9, 100)
    
    # 1. Sorting algorithms
    ax = axes[0, 0]
    ax.set_title('Sorting Algorithms')
    
    # QuickSort (in-place)
    ax.loglog(n_values * 0 + 1, n_values * np.log2(n_values), 
             label='QuickSort (O(1) space)', linewidth=2)
    
    # MergeSort (standard)
    ax.loglog(n_values, n_values * np.log2(n_values), 
             label='MergeSort (O(n) space)', linewidth=2)
    
    # External MergeSort (√n buffers)
    ax.loglog(np.sqrt(n_values), n_values * np.log2(n_values) * 2, 
             label='External Sort (O(√n) space)', linewidth=2)
    
    ax.set_xlabel('Space Usage')
    ax.set_ylabel('Time Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Search structures
    ax = axes[0, 1]
    ax.set_title('Search Data Structures')
    
    # Array (unsorted)
    ax.loglog(n_values, n_values, 
             label='Array Search (O(n) time)', linewidth=2)
    
    # Binary Search Tree
    ax.loglog(n_values, np.log2(n_values), 
             label='BST (O(log n) average)', linewidth=2)
    
    # Hash Table
    ax.loglog(n_values, n_values * 0 + 1, 
             label='Hash Table (O(1) average)', linewidth=2)
    
    # B-tree (√n fanout)
    ax.loglog(n_values, np.log(n_values) / np.log(np.sqrt(n_values)), 
             label='B-tree (O(log_√n n))', linewidth=2)
    
    ax.set_xlabel('Space Usage')
    ax.set_ylabel('Search Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Matrix operations
    ax = axes[1, 0]
    ax.set_title('Matrix Multiplication')
    
    n_matrix = np.sqrt(n_values)  # Matrix dimension
    
    # Standard multiplication
    ax.loglog(n_matrix**2, n_matrix**3, 
             label='Standard (O(n²) space)', linewidth=2)
    
    # Strassen's algorithm
    ax.loglog(n_matrix**2, n_matrix**2.807, 
             label='Strassen (O(n²) space)', linewidth=2)
    
    # Block multiplication (√n blocks)
    ax.loglog(n_matrix**1.5, n_matrix**3 * 1.2, 
             label='Blocked (O(n^1.5) space)', linewidth=2)
    
    ax.set_xlabel('Space Usage')
    ax.set_ylabel('Time Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Graph algorithms
    ax = axes[1, 1]
    ax.set_title('Graph Algorithms')
    
    # BFS/DFS
    ax.loglog(n_values, n_values + n_values, 
             label='BFS/DFS (O(V+E) space)', linewidth=2)
    
    # Dijkstra
    ax.loglog(n_values * np.log(n_values), n_values * np.log(n_values), 
             label='Dijkstra (O(V log V) space)', linewidth=2)
    
    # A* with bounded memory
    ax.loglog(np.sqrt(n_values), n_values * np.sqrt(n_values), 
             label='Memory-bounded A* (O(√V) space)', linewidth=2)
    
    ax.set_xlabel('Space Usage')
    ax.set_ylabel('Time Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_real_world_systems():
    """Visualize real-world system tradeoffs"""
    print("\n" + "="*60)
    print("Real-World System Tradeoffs")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Space-Time Tradeoffs in Production Systems', fontsize=16)
    
    # 1. Database systems
    ax = axes[0, 0]
    ax.set_title('Database Buffer Pool Strategies')
    
    data_sizes = np.logspace(6, 12, 50)  # 1MB to 1TB
    memory_sizes = [8e9, 32e9, 128e9]  # 8GB, 32GB, 128GB RAM
    
    for mem in memory_sizes:
        # Full caching
        full_cache_perf = np.minimum(data_sizes / mem, 1.0)
        
        # √n caching
        sqrt_cache_size = np.sqrt(data_sizes)
        sqrt_cache_perf = np.minimum(sqrt_cache_size / mem, 1.0) * 0.9
        
        ax.semilogx(data_sizes / 1e9, full_cache_perf, 
                   label=f'Full cache ({mem/1e9:.0f}GB RAM)', linewidth=2)
        ax.semilogx(data_sizes / 1e9, sqrt_cache_perf, '--',
                   label=f'√n cache ({mem/1e9:.0f}GB RAM)', linewidth=2)
    
    ax.set_xlabel('Database Size (GB)')
    ax.set_ylabel('Cache Hit Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. LLM inference
    ax = axes[0, 1]
    ax.set_title('LLM Inference: KV-Cache Strategies')
    
    sequence_lengths = np.logspace(1, 5, 50)  # 10 to 100K tokens
    
    # Full KV-cache
    full_memory = sequence_lengths * 2048 * 4 * 2  # seq * dim * float32 * KV
    full_speed = sequence_lengths * 0 + 200  # tokens/sec
    
    # Flash Attention (√n memory)
    flash_memory = np.sqrt(sequence_lengths) * 2048 * 4 * 2
    flash_speed = 180 - sequence_lengths / 1000  # Slight slowdown
    
    # Paged Attention
    paged_memory = sequence_lengths * 2048 * 4 * 2 * 0.1  # 10% of full
    paged_speed = 150 - sequence_lengths / 500
    
    ax2 = ax.twinx()
    
    l1 = ax.loglog(sequence_lengths, full_memory / 1e9, 'b-', 
                   label='Full KV-cache (memory)', linewidth=2)
    l2 = ax.loglog(sequence_lengths, flash_memory / 1e9, 'r-', 
                   label='Flash Attention (memory)', linewidth=2)
    l3 = ax.loglog(sequence_lengths, paged_memory / 1e9, 'g-', 
                   label='Paged Attention (memory)', linewidth=2)
    
    l4 = ax2.semilogx(sequence_lengths, full_speed, 'b--', 
                      label='Full KV-cache (speed)', linewidth=2)
    l5 = ax2.semilogx(sequence_lengths, flash_speed, 'r--', 
                      label='Flash Attention (speed)', linewidth=2)
    l6 = ax2.semilogx(sequence_lengths, paged_speed, 'g--', 
                      label='Paged Attention (speed)', linewidth=2)
    
    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Memory Usage (GB)')
    ax2.set_ylabel('Inference Speed (tokens/sec)')
    
    # Combine legends
    lns = l1 + l2 + l3 + l4 + l5 + l6
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    # 3. Distributed computing
    ax = axes[1, 0]
    ax.set_title('MapReduce Shuffle Strategies')
    
    data_per_node = np.logspace(6, 11, 50)  # 1MB to 100GB per node
    num_nodes = 100
    
    # All-to-all shuffle
    all_to_all_mem = data_per_node * num_nodes
    all_to_all_time = data_per_node * num_nodes / 1e9  # Network time
    
    # Tree aggregation (√n levels)
    tree_levels = int(np.sqrt(num_nodes))
    tree_mem = data_per_node * tree_levels
    tree_time = data_per_node * tree_levels / 1e9
    
    # Combiner optimization
    combiner_mem = data_per_node * np.log2(num_nodes)
    combiner_time = data_per_node * np.log2(num_nodes) / 1e9
    
    ax.loglog(all_to_all_mem / 1e9, all_to_all_time, 
             label='All-to-all shuffle', linewidth=2)
    ax.loglog(tree_mem / 1e9, tree_time, 
             label='Tree aggregation (√n)', linewidth=2)
    ax.loglog(combiner_mem / 1e9, combiner_time, 
             label='With combiners', linewidth=2)
    
    ax.set_xlabel('Memory per Node (GB)')
    ax.set_ylabel('Shuffle Time (seconds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Mobile/embedded systems
    ax = axes[1, 1]
    ax.set_title('Mobile App Memory Strategies')
    
    image_counts = np.logspace(1, 4, 50)  # 10 to 10K images
    image_size = 2e6  # 2MB per image
    
    # Full cache
    full_cache = image_counts * image_size / 1e9
    full_load_time = image_counts * 0 + 0.1  # Instant from cache
    
    # LRU cache (√n size)
    lru_cache = np.sqrt(image_counts) * image_size / 1e9
    lru_load_time = 0.1 + (1 - np.sqrt(image_counts) / image_counts) * 2
    
    # No cache
    no_cache = image_counts * 0 + 0.01  # Minimal memory
    no_load_time = image_counts * 0 + 2  # Always load from network
    
    ax2 = ax.twinx()
    
    l1 = ax.loglog(image_counts, full_cache, 'b-', 
                   label='Full cache (memory)', linewidth=2)
    l2 = ax.loglog(image_counts, lru_cache, 'r-', 
                   label='√n LRU cache (memory)', linewidth=2)
    l3 = ax.loglog(image_counts, no_cache, 'g-', 
                   label='No cache (memory)', linewidth=2)
    
    l4 = ax2.semilogx(image_counts, full_load_time, 'b--', 
                      label='Full cache (load time)', linewidth=2)
    l5 = ax2.semilogx(image_counts, lru_load_time, 'r--', 
                      label='√n LRU cache (load time)', linewidth=2)
    l6 = ax2.semilogx(image_counts, no_load_time, 'g--', 
                      label='No cache (load time)', linewidth=2)
    
    ax.set_xlabel('Number of Images')
    ax.set_ylabel('Memory Usage (GB)')
    ax2.set_ylabel('Average Load Time (seconds)')
    
    # Combine legends
    lns = l1 + l2 + l3 + l4 + l5 + l6
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_optimization_impact():
    """Show impact of √n optimizations"""
    print("\n" + "="*60)
    print("Impact of √n Optimizations")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memory Savings and Performance Impact', fontsize=16)
    
    # Common data sizes
    n_values = np.logspace(3, 12, 50)
    
    # 1. Memory savings
    ax = axes[0, 0]
    ax.set_title('Memory Reduction Factor')
    
    reduction_factor = n_values / np.sqrt(n_values)
    
    ax.loglog(n_values, reduction_factor, 'b-', linewidth=3)
    
    # Add markers for common sizes
    common_sizes = [1e3, 1e6, 1e9, 1e12]
    common_names = ['1K', '1M', '1B', '1T']
    
    for size, name in zip(common_sizes, common_names):
        factor = size / np.sqrt(size)
        ax.scatter(size, factor, s=100, zorder=5)
        ax.annotate(f'{name}: {factor:.0f}x', 
                   xy=(size, factor), 
                   xytext=(size*2, factor*1.5),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_xlabel('Data Size (n)')
    ax.set_ylabel('Memory Reduction (n/√n)')
    ax.grid(True, alpha=0.3)
    
    # 2. Time overhead
    ax = axes[0, 1]
    ax.set_title('Time Overhead of √n Strategies')
    
    # Different overhead scenarios
    low_overhead = np.ones_like(n_values) * 1.1  # 10% overhead
    medium_overhead = 1 + np.log10(n_values) / 10  # Logarithmic growth
    high_overhead = 1 + np.sqrt(n_values) / n_values * 100  # Diminishing
    
    ax.semilogx(n_values, low_overhead, label='Low overhead (10%)', linewidth=2)
    ax.semilogx(n_values, medium_overhead, label='Medium overhead', linewidth=2)
    ax.semilogx(n_values, high_overhead, label='High overhead', linewidth=2)
    
    ax.axhline(y=2, color='red', linestyle='--', label='2x slowdown limit')
    
    ax.set_xlabel('Data Size (n)')
    ax.set_ylabel('Time Overhead Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Cost efficiency
    ax = axes[1, 0]
    ax.set_title('Cloud Cost Analysis')
    
    # Cost model: memory cost + compute cost
    memory_cost_per_gb = 0.1  # $/GB/hour
    compute_cost_per_cpu = 0.05  # $/CPU/hour
    
    # Standard approach
    standard_memory_cost = n_values / 1e9 * memory_cost_per_gb
    standard_compute_cost = np.ones_like(n_values) * compute_cost_per_cpu
    standard_total = standard_memory_cost + standard_compute_cost
    
    # √n approach
    sqrt_memory_cost = np.sqrt(n_values) / 1e9 * memory_cost_per_gb
    sqrt_compute_cost = np.ones_like(n_values) * compute_cost_per_cpu * 1.2
    sqrt_total = sqrt_memory_cost + sqrt_compute_cost
    
    ax.loglog(n_values, standard_total, label='Standard (O(n) memory)', linewidth=2)
    ax.loglog(n_values, sqrt_total, label='√n optimized', linewidth=2)
    
    # Savings region
    ax.fill_between(n_values, sqrt_total, standard_total, 
                   where=(standard_total > sqrt_total),
                   alpha=0.3, color='green', label='Cost savings')
    
    ax.set_xlabel('Data Size (bytes)')
    ax.set_ylabel('Cost ($/hour)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Breakeven analysis
    ax = axes[1, 1]
    ax.set_title('When to Use √n Optimizations')
    
    # Create a heatmap showing when √n is beneficial
    data_sizes = np.logspace(3, 9, 20)
    memory_costs = np.logspace(-2, 2, 20)
    
    benefit_matrix = np.zeros((len(memory_costs), len(data_sizes)))
    
    for i, mem_cost in enumerate(memory_costs):
        for j, data_size in enumerate(data_sizes):
            # Simple model: benefit if memory savings > compute overhead
            memory_saved = (data_size - np.sqrt(data_size)) / 1e9
            benefit = memory_saved * mem_cost - 0.1  # 0.1 = overhead cost
            benefit_matrix[i, j] = benefit > 0
    
    im = ax.imshow(benefit_matrix, aspect='auto', origin='lower',
                   extent=[3, 9, -2, 2], cmap='RdYlGn')
    
    ax.set_xlabel('log₁₀(Data Size)')
    ax.set_ylabel('log₁₀(Memory Cost Ratio)')
    ax.set_title('Green = Use √n, Red = Use Standard')
    
    # Add contour line
    contour = ax.contour(np.log10(data_sizes), np.log10(memory_costs),
                        benefit_matrix, levels=[0.5], colors='black', linewidths=2)
    ax.clabel(contour, inline=True, fmt='Breakeven')
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()


def create_educational_diagrams():
    """Create educational diagrams explaining concepts"""
    print("\n" + "="*60)
    print("Educational Diagrams")
    print("="*60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Williams' theorem visualization
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_title("Williams' Space-Time Bound", fontsize=14, fontweight='bold')
    
    t_values = np.logspace(1, 6, 100)
    s_bound = np.sqrt(t_values * np.log(t_values))
    
    ax1.fill_between(t_values, 0, s_bound, alpha=0.3, color='red', 
                    label='Impossible region')
    ax1.fill_between(t_values, s_bound, t_values*10, alpha=0.3, color='green',
                    label='Feasible region')
    ax1.loglog(t_values, s_bound, 'k-', linewidth=3, 
              label='S = √(t log t) bound')
    
    # Add example algorithms
    ax1.scatter([1000], [1000], s=100, color='blue', marker='o',
               label='Standard algorithm')
    ax1.scatter([1000], [31.6], s=100, color='orange', marker='s',
               label='√n algorithm')
    
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('Space (s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory hierarchy
    ax2 = plt.subplot(2, 3, 2)
    ax2.set_title('Memory Hierarchy & Access Times', fontsize=14, fontweight='bold')
    
    levels = ['CPU\nRegisters', 'L1\nCache', 'L2\nCache', 'L3\nCache', 'RAM', 'SSD', 'HDD']
    sizes = [1e-3, 32, 256, 8192, 32768, 512000, 2000000]  # KB
    latencies = [0.3, 1, 3, 12, 100, 10000, 10000000]  # ns
    
    y_pos = np.arange(len(levels))
    
    # Create bars
    bars = ax2.barh(y_pos, np.log10(sizes), color=plt.cm.viridis(np.linspace(0, 1, len(levels))))
    
    # Add latency annotations
    for i, (bar, latency) in enumerate(zip(bars, latencies)):
        width = bar.get_width()
        if latency < 1000:
            lat_str = f'{latency:.1f}ns'
        elif latency < 1000000:
            lat_str = f'{latency/1000:.0f}μs'
        else:
            lat_str = f'{latency/1000000:.0f}ms'
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                lat_str, va='center')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(levels)
    ax2.set_xlabel('log₁₀(Size in KB)')
    ax2.set_title('Memory Hierarchy & Access Times', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Checkpoint visualization
    ax3 = plt.subplot(2, 3, 3)
    ax3.set_title('Checkpoint Strategies', fontsize=14, fontweight='bold')
    
    n = 100
    progress = np.arange(n)
    
    # No checkpointing
    ax3.fill_between(progress, 0, progress, alpha=0.3, color='red', 
                    label='No checkpoint')
    
    # √n checkpointing
    checkpoint_interval = int(np.sqrt(n))
    sqrt_memory = np.zeros(n)
    for i in range(n):
        sqrt_memory[i] = i % checkpoint_interval
    ax3.fill_between(progress, 0, sqrt_memory, alpha=0.3, color='green',
                    label='√n checkpoint')
    
    # Fixed interval
    fixed_interval = 20
    fixed_memory = np.zeros(n)
    for i in range(n):
        fixed_memory[i] = i % fixed_interval
    ax3.plot(progress, fixed_memory, 'b-', linewidth=2, 
            label=f'Fixed interval ({fixed_interval})')
    
    # Add checkpoint markers
    for i in range(0, n, checkpoint_interval):
        ax3.axvline(x=i, color='green', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Progress')
    ax3.set_ylabel('Memory Usage')
    ax3.legend()
    ax3.set_xlim(0, n)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cache line utilization
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_title('Cache Line Utilization', fontsize=14, fontweight='bold')
    
    cache_line_size = 64  # bytes
    
    # Poor alignment
    poor_sizes = [7, 13, 17, 23]  # bytes per element
    poor_util = [cache_line_size // s * s / cache_line_size * 100 for s in poor_sizes]
    
    # Good alignment  
    good_sizes = [8, 16, 32, 64]  # bytes per element
    good_util = [cache_line_size // s * s / cache_line_size * 100 for s in good_sizes]
    
    x = np.arange(len(poor_sizes))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, poor_util, width, label='Poor alignment', color='red', alpha=0.7)
    bars2 = ax4.bar(x + width/2, good_util, width, label='Good alignment', color='green', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom')
    
    ax4.set_ylabel('Cache Line Utilization (%)')
    ax4.set_xlabel('Element Size Configuration')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{p}B vs {g}B' for p, g in zip(poor_sizes, good_sizes)])
    ax4.legend()
    ax4.set_ylim(0, 110)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Algorithm selection guide
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_title('Algorithm Selection Guide', fontsize=14, fontweight='bold')
    
    # Create decision matrix
    data_size_ranges = ['< 1KB', '1KB-1MB', '1MB-1GB', '> 1GB']
    memory_constraints = ['Unlimited', 'Limited', 'Severe', 'Embedded']
    
    recommendations = [
        ['Array', 'Array', 'Hash', 'B-tree'],
        ['Array', 'B-tree', 'B-tree', 'External'],
        ['Compressed', 'Compressed', '√n Cache', '√n External'],
        ['Minimal', 'Minimal', 'Streaming', 'Streaming']
    ]
    
    # Create color map
    colors = {'Array': 0, 'Hash': 1, 'B-tree': 2, 'External': 3,
             'Compressed': 4, '√n Cache': 5, '√n External': 6,
             'Minimal': 7, 'Streaming': 8}
    
    matrix = np.zeros((len(memory_constraints), len(data_size_ranges)))
    
    for i in range(len(memory_constraints)):
        for j in range(len(data_size_ranges)):
            matrix[i, j] = colors[recommendations[i][j]]
    
    im = ax5.imshow(matrix, cmap='tab10', aspect='auto')
    
    # Add text annotations
    for i in range(len(memory_constraints)):
        for j in range(len(data_size_ranges)):
            ax5.text(j, i, recommendations[i][j], 
                    ha='center', va='center', fontsize=10)
    
    ax5.set_xticks(np.arange(len(data_size_ranges)))
    ax5.set_yticks(np.arange(len(memory_constraints)))
    ax5.set_xticklabels(data_size_ranges)
    ax5.set_yticklabels(memory_constraints)
    ax5.set_xlabel('Data Size')
    ax5.set_ylabel('Memory Constraint')
    
    # 6. Cost-benefit analysis
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_title('Cost-Benefit Analysis', fontsize=14, fontweight='bold')
    
    # Create spider chart
    categories = ['Memory\nSavings', 'Speed', 'Complexity', 'Fault\nTolerance', 'Scalability']
    
    # Different strategies
    strategies = {
        'Standard': [20, 100, 100, 30, 40],
        '√n Optimized': [90, 70, 60, 80, 95],
        'Extreme Memory': [98, 30, 20, 50, 80]
    }
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    for name, values in strategies.items():
        values += values[:1]  # Complete the circle
        ax6.plot(angles, values, 'o-', linewidth=2, label=name)
        ax6.fill(angles, values, alpha=0.15)
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 100)
    ax6.set_title('Strategy Comparison', fontsize=14, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all example visualizations"""
    print("SpaceTime Explorer - Example Visualizations")
    print("="*60)
    
    # Run each visualization
    visualize_algorithm_comparison()
    visualize_real_world_systems()
    visualize_optimization_impact()
    create_educational_diagrams()
    
    print("\n" + "="*60)
    print("Example visualizations complete!")
    print("\nThese examples demonstrate:")
    print("- Algorithm space-time tradeoffs")
    print("- Real-world system optimizations")
    print("- Impact of √n strategies")
    print("- Educational diagrams for understanding concepts")
    print("="*60)


if __name__ == "__main__":
    main()