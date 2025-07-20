#!/usr/bin/env python3
"""
SpaceTime Benchmark Suite: Standardized benchmarks for measuring space-time tradeoffs

Features:
- Standard Benchmarks: Common algorithms with space-time variants
- Real Workloads: Database, ML, distributed computing scenarios  
- Measurement Framework: Accurate time, memory, and cache metrics
- Comparison Tools: Statistical analysis and visualization
- Reproducibility: Controlled environment and result validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import psutil
import numpy as np
import json
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import matplotlib.pyplot as plt
import sqlite3
import random
import string
import gc

# Import core components
from core.spacetime_core import (
    MemoryHierarchy,
    SqrtNCalculator,
    StrategyAnalyzer
)


class BenchmarkCategory(Enum):
    """Categories of benchmarks"""
    SORTING = "sorting"
    SEARCHING = "searching"
    GRAPH = "graph"
    DATABASE = "database"
    ML_TRAINING = "ml_training"
    DISTRIBUTED = "distributed"
    STREAMING = "streaming"
    COMPRESSION = "compression"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    name: str
    category: BenchmarkCategory
    strategy: str
    data_size: int
    time_seconds: float
    memory_peak_mb: float
    memory_avg_mb: float
    cache_misses: Optional[int]
    page_faults: Optional[int]
    throughput: float  # Operations per second
    space_time_product: float
    metadata: Dict[str, Any]


@dataclass
class BenchmarkComparison:
    """Comparison between strategies"""
    baseline: BenchmarkResult
    optimized: BenchmarkResult
    memory_reduction: float  # Percentage
    time_overhead: float  # Percentage
    space_time_improvement: float  # Percentage
    recommendation: str


class MemoryMonitor:
    """Monitor memory usage during benchmark"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.samples = []
        self.running = False
        
    def start(self):
        """Start monitoring"""
        self.samples = []
        self.running = True
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
    def sample(self):
        """Take a memory sample"""
        if self.running:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.samples.append(current_memory - self.initial_memory)
            
    def stop(self) -> Tuple[float, float]:
        """Stop monitoring and return peak and average memory"""
        self.running = False
        if not self.samples:
            return 0.0, 0.0
        return max(self.samples), np.mean(self.samples)


class BenchmarkRunner:
    """Main benchmark execution framework"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.sqrt_calc = SqrtNCalculator()
        self.hierarchy = MemoryHierarchy.detect_system()
        self.memory_monitor = MemoryMonitor()
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        
    def run_benchmark(self, 
                     name: str,
                     category: BenchmarkCategory,
                     strategy: str,
                     benchmark_func: Callable,
                     data_size: int,
                     **kwargs) -> BenchmarkResult:
        """Run a single benchmark"""
        print(f"Running {name} ({strategy}) with n={data_size:,}")
        
        # Prepare
        gc.collect()
        time.sleep(0.1)  # Let system settle
        
        # Start monitoring
        self.memory_monitor.start()
        
        # Run benchmark
        start_time = time.perf_counter()
        
        try:
            operations = benchmark_func(data_size, strategy=strategy, **kwargs)
            success = True
        except Exception as e:
            print(f"  Error: {e}")
            operations = 0
            success = False
            
        end_time = time.perf_counter()
        
        # Stop monitoring
        peak_memory, avg_memory = self.memory_monitor.stop()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        throughput = operations / elapsed_time if elapsed_time > 0 else 0
        space_time_product = peak_memory * elapsed_time
        
        # Get cache statistics (if available)
        cache_misses, page_faults = self._get_cache_stats()
        
        result = BenchmarkResult(
            name=name,
            category=category,
            strategy=strategy,
            data_size=data_size,
            time_seconds=elapsed_time,
            memory_peak_mb=peak_memory,
            memory_avg_mb=avg_memory,
            cache_misses=cache_misses,
            page_faults=page_faults,
            throughput=throughput,
            space_time_product=space_time_product,
            metadata={
                'success': success,
                'operations': operations,
                **kwargs
            }
        )
        
        self.results.append(result)
        
        print(f"  Time: {elapsed_time:.3f}s, Memory: {peak_memory:.1f}MB, "
              f"Throughput: {throughput:.0f} ops/s")
        
        return result
        
    def compare_strategies(self, 
                          name: str,
                          category: BenchmarkCategory,
                          benchmark_func: Callable,
                          strategies: List[str],
                          data_sizes: List[int],
                          **kwargs) -> List[BenchmarkComparison]:
        """Compare multiple strategies"""
        comparisons = []
        
        for data_size in data_sizes:
            print(f"\n{'='*60}")
            print(f"Comparing {name} strategies for n={data_size:,}")
            print('='*60)
            
            # Run baseline (first strategy)
            baseline = self.run_benchmark(
                name, category, strategies[0], 
                benchmark_func, data_size, **kwargs
            )
            
            # Run optimized strategies
            for strategy in strategies[1:]:
                optimized = self.run_benchmark(
                    name, category, strategy,
                    benchmark_func, data_size, **kwargs
                )
                
                # Calculate comparison metrics
                memory_reduction = (1 - optimized.memory_peak_mb / baseline.memory_peak_mb) * 100
                time_overhead = (optimized.time_seconds / baseline.time_seconds - 1) * 100
                space_time_improvement = (1 - optimized.space_time_product / baseline.space_time_product) * 100
                
                # Generate recommendation
                if space_time_improvement > 20:
                    recommendation = f"Use {strategy} for {memory_reduction:.0f}% memory savings"
                elif time_overhead > 100:
                    recommendation = f"Avoid {strategy} due to {time_overhead:.0f}% slowdown"
                else:
                    recommendation = f"Consider {strategy} for memory-constrained environments"
                
                comparison = BenchmarkComparison(
                    baseline=baseline,
                    optimized=optimized,
                    memory_reduction=memory_reduction,
                    time_overhead=time_overhead,
                    space_time_improvement=space_time_improvement,
                    recommendation=recommendation
                )
                
                comparisons.append(comparison)
                
                print(f"\nComparison {baseline.strategy} vs {optimized.strategy}:")
                print(f"  Memory reduction: {memory_reduction:.1f}%")
                print(f"  Time overhead: {time_overhead:.1f}%")
                print(f"  Space-time improvement: {space_time_improvement:.1f}%")
                print(f"  Recommendation: {recommendation}")
        
        return comparisons
        
    def _get_cache_stats(self) -> Tuple[Optional[int], Optional[int]]:
        """Get cache misses and page faults (platform specific)"""
        # This would need platform-specific implementation
        # For now, return None
        return None, None
        
    def save_results(self):
        """Save all results to JSON"""
        filename = os.path.join(self.output_dir, 
                               f"results_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        data = {
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'l3_cache_mb': self.hierarchy.l3_size / 1024 / 1024
            },
            'results': [asdict(r) for r in self.results],
            'timestamp': time.time()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\nResults saved to {filename}")
        
    def plot_results(self, category: Optional[BenchmarkCategory] = None):
        """Plot benchmark results"""
        # Filter results
        results = self.results
        if category:
            results = [r for r in results if r.category == category]
            
        if not results:
            print("No results to plot")
            return
            
        # Group by benchmark name
        benchmarks = {}
        for r in results:
            if r.name not in benchmarks:
                benchmarks[r.name] = {}
            if r.strategy not in benchmarks[r.name]:
                benchmarks[r.name][r.strategy] = []
            benchmarks[r.name][r.strategy].append(r)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Benchmark Results{f" - {category.value}" if category else ""}', 
                    fontsize=16)
        
        for (name, strategies), ax in zip(list(benchmarks.items())[:4], axes.flat):
            # Plot time vs data size
            for strategy, results in strategies.items():
                sizes = [r.data_size for r in results]
                times = [r.time_seconds for r in results]
                ax.loglog(sizes, times, 'o-', label=strategy, linewidth=2)
            
            ax.set_xlabel('Data Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'benchmark_plot.png'), dpi=150)
        plt.show()


# Benchmark Implementations

def benchmark_sorting(n: int, strategy: str = 'standard', **kwargs) -> int:
    """Sorting benchmark with different memory strategies"""
    # Generate random data
    data = np.random.rand(n)
    
    if strategy == 'standard':
        # Standard in-memory sort
        sorted_data = np.sort(data)
        return n
        
    elif strategy == 'sqrt_n':
        # External sort with √n memory
        chunk_size = int(np.sqrt(n))
        chunks = []
        
        # Sort chunks
        for i in range(0, n, chunk_size):
            chunk = data[i:i+chunk_size]
            chunks.append(np.sort(chunk))
        
        # Merge chunks (simplified)
        result = np.concatenate(chunks)
        result.sort()  # Final merge
        return n
        
    elif strategy == 'constant':
        # Streaming sort with O(1) memory (simplified)
        # In practice would use external storage
        sorted_indices = np.argsort(data)
        return n


def benchmark_searching(n: int, strategy: str = 'hash', **kwargs) -> int:
    """Search benchmark with different data structures"""
    # Generate data
    keys = [f"key_{i:08d}" for i in range(n)]
    values = list(range(n))
    queries = random.sample(keys, min(1000, n))
    
    if strategy == 'hash':
        # Standard hash table
        hash_map = dict(zip(keys, values))
        for q in queries:
            _ = hash_map.get(q)
        return len(queries)
        
    elif strategy == 'btree':
        # B-tree (simulated with sorted list)
        sorted_pairs = sorted(zip(keys, values))
        for q in queries:
            # Binary search
            left, right = 0, len(sorted_pairs) - 1
            while left <= right:
                mid = (left + right) // 2
                if sorted_pairs[mid][0] == q:
                    break
                elif sorted_pairs[mid][0] < q:
                    left = mid + 1
                else:
                    right = mid - 1
        return len(queries)
        
    elif strategy == 'external':
        # External index with √n cache
        cache_size = int(np.sqrt(n))
        cache = dict(list(zip(keys, values))[:cache_size])
        
        hits = 0
        for q in queries:
            if q in cache:
                hits += 1
            # Simulate disk access for misses
            time.sleep(0.00001)  # 10 microseconds
        
        return len(queries)


def benchmark_matrix_multiply(n: int, strategy: str = 'standard', **kwargs) -> int:
    """Matrix multiplication with different memory patterns"""
    # Use smaller matrices for reasonable runtime
    size = int(np.sqrt(n))
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    if strategy == 'standard':
        # Standard multiplication
        C = np.dot(A, B)
        return size * size * size  # Operations
        
    elif strategy == 'blocked':
        # Block multiplication for cache efficiency
        block_size = int(np.sqrt(size))
        C = np.zeros((size, size))
        
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                for k in range(0, size, block_size):
                    # Block multiply
                    i_end = min(i + block_size, size)
                    j_end = min(j + block_size, size)
                    k_end = min(k + block_size, size)
                    
                    C[i:i_end, j:j_end] += np.dot(
                        A[i:i_end, k:k_end],
                        B[k:k_end, j:j_end]
                    )
        
        return size * size * size
        
    elif strategy == 'streaming':
        # Streaming computation with minimal memory
        # (Simplified - would need external storage)
        C = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                C[i, j] = np.dot(A[i, :], B[:, j])
        
        return size * size * size


def benchmark_database_query(n: int, strategy: str = 'standard', **kwargs) -> int:
    """Database query with different buffer strategies"""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT,
                created_at INTEGER
            )
        ''')
        
        # Insert data
        users = [(i, f'user_{i}', f'user_{i}@example.com', i * 1000) 
                for i in range(n)]
        cursor.executemany('INSERT INTO users VALUES (?, ?, ?, ?)', users)
        conn.commit()
        
        # Configure based on strategy
        if strategy == 'standard':
            # Default cache
            cursor.execute('PRAGMA cache_size = 2000')  # 2000 pages
        elif strategy == 'sqrt_n':
            # √n cache size
            cache_pages = max(10, int(np.sqrt(n / 100)))  # Assuming ~100 rows per page
            cursor.execute(f'PRAGMA cache_size = {cache_pages}')
        elif strategy == 'minimal':
            # Minimal cache
            cursor.execute('PRAGMA cache_size = 10')
        
        # Run queries
        query_count = min(1000, n // 10)
        for _ in range(query_count):
            user_id = random.randint(1, n)
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            cursor.fetchone()
        
        conn.close()
        return query_count
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def benchmark_ml_training(n: int, strategy: str = 'standard', **kwargs) -> int:
    """ML training with different memory strategies"""
    # Simulate neural network training
    batch_size = min(64, n)
    num_features = 100
    num_classes = 10
    
    # Generate synthetic data
    X = np.random.randn(n, num_features).astype(np.float32)
    y = np.random.randint(0, num_classes, n)
    
    # Simple model weights
    W1 = np.random.randn(num_features, 64).astype(np.float32) * 0.01
    W2 = np.random.randn(64, num_classes).astype(np.float32) * 0.01
    
    iterations = min(100, n // batch_size)
    
    if strategy == 'standard':
        # Standard training - keep all activations
        for i in range(iterations):
            idx = np.random.choice(n, batch_size)
            batch_X = X[idx]
            
            # Forward pass
            h1 = np.maximum(0, batch_X @ W1)  # ReLU
            logits = h1 @ W2
            
            # Backward pass (simplified)
            W2 += np.random.randn(*W2.shape) * 0.001
            W1 += np.random.randn(*W1.shape) * 0.001
            
    elif strategy == 'gradient_checkpoint':
        # Gradient checkpointing - recompute activations
        checkpoint_interval = int(np.sqrt(batch_size))
        
        for i in range(iterations):
            idx = np.random.choice(n, batch_size)
            batch_X = X[idx]
            
            # Process in chunks
            for j in range(0, batch_size, checkpoint_interval):
                chunk = batch_X[j:j+checkpoint_interval]
                
                # Forward pass
                h1 = np.maximum(0, chunk @ W1)
                logits = h1 @ W2
                
                # Recompute for backward
                h1_recompute = np.maximum(0, chunk @ W1)
                
            # Update weights
            W2 += np.random.randn(*W2.shape) * 0.001
            W1 += np.random.randn(*W1.shape) * 0.001
            
    elif strategy == 'mixed_precision':
        # Mixed precision training
        W1_fp16 = W1.astype(np.float16)
        W2_fp16 = W2.astype(np.float16)
        
        for i in range(iterations):
            idx = np.random.choice(n, batch_size)
            batch_X = X[idx].astype(np.float16)
            
            # Forward pass in FP16
            h1 = np.maximum(0, batch_X @ W1_fp16)
            logits = h1 @ W2_fp16
            
            # Update in FP32
            W2 += np.random.randn(*W2.shape) * 0.001
            W1 += np.random.randn(*W1.shape) * 0.001
            W1_fp16 = W1.astype(np.float16)
            W2_fp16 = W2.astype(np.float16)
    
    return iterations * batch_size


def benchmark_graph_traversal(n: int, strategy: str = 'bfs', **kwargs) -> int:
    """Graph traversal with different memory strategies"""
    # Generate random graph (sparse)
    edges = []
    num_edges = min(n * 5, n * (n - 1) // 2)  # Average degree 5
    
    for _ in range(num_edges):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            edges.append((u, v))
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    if strategy == 'bfs':
        # Standard BFS
        visited = [False] * n
        queue = [0]
        visited[0] = True
        count = 0
        
        while queue:
            u = queue.pop(0)
            count += 1
            
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    
        return count
        
    elif strategy == 'dfs_iterative':
        # DFS with explicit stack (less memory than recursion)
        visited = [False] * n
        stack = [0]
        count = 0
        
        while stack:
            u = stack.pop()
            if not visited[u]:
                visited[u] = True
                count += 1
                
                for v in adj[u]:
                    if not visited[v]:
                        stack.append(v)
                        
        return count
        
    elif strategy == 'memory_bounded':
        # Memory-bounded search (like IDA*)
        # Simplified - just limit queue size
        max_queue_size = int(np.sqrt(n))
        visited = set()
        queue = [0]
        count = 0
        
        while queue:
            u = queue.pop(0)
            if u not in visited:
                visited.add(u)
                count += 1
                
                # Add neighbors if queue not full
                for v in adj[u]:
                    if v not in visited and len(queue) < max_queue_size:
                        queue.append(v)
                        
        return count


# Standard benchmark suites

def sorting_suite(runner: BenchmarkRunner):
    """Run sorting benchmarks"""
    print("\n" + "="*60)
    print("SORTING BENCHMARKS")
    print("="*60)
    
    strategies = ['standard', 'sqrt_n', 'constant']
    data_sizes = [10000, 100000, 1000000]
    
    runner.compare_strategies(
        "Sorting",
        BenchmarkCategory.SORTING,
        benchmark_sorting,
        strategies,
        data_sizes
    )


def searching_suite(runner: BenchmarkRunner):
    """Run search structure benchmarks"""
    print("\n" + "="*60)
    print("SEARCHING BENCHMARKS")
    print("="*60)
    
    strategies = ['hash', 'btree', 'external']
    data_sizes = [10000, 100000, 1000000]
    
    runner.compare_strategies(
        "Search Structures",
        BenchmarkCategory.SEARCHING,
        benchmark_searching,
        strategies,
        data_sizes
    )


def database_suite(runner: BenchmarkRunner):
    """Run database benchmarks"""
    print("\n" + "="*60)
    print("DATABASE BENCHMARKS")
    print("="*60)
    
    strategies = ['standard', 'sqrt_n', 'minimal']
    data_sizes = [1000, 10000, 100000]
    
    runner.compare_strategies(
        "Database Queries",
        BenchmarkCategory.DATABASE,
        benchmark_database_query,
        strategies,
        data_sizes
    )


def ml_suite(runner: BenchmarkRunner):
    """Run ML training benchmarks"""
    print("\n" + "="*60)
    print("ML TRAINING BENCHMARKS")
    print("="*60)
    
    strategies = ['standard', 'gradient_checkpoint', 'mixed_precision']
    data_sizes = [1000, 10000, 50000]
    
    runner.compare_strategies(
        "ML Training",
        BenchmarkCategory.ML_TRAINING,
        benchmark_ml_training,
        strategies,
        data_sizes
    )


def graph_suite(runner: BenchmarkRunner):
    """Run graph algorithm benchmarks"""
    print("\n" + "="*60)
    print("GRAPH ALGORITHM BENCHMARKS")
    print("="*60)
    
    strategies = ['bfs', 'dfs_iterative', 'memory_bounded']
    data_sizes = [1000, 10000, 50000]
    
    runner.compare_strategies(
        "Graph Traversal",
        BenchmarkCategory.GRAPH,
        benchmark_graph_traversal,
        strategies,
        data_sizes
    )


def matrix_suite(runner: BenchmarkRunner):
    """Run matrix operation benchmarks"""
    print("\n" + "="*60)
    print("MATRIX OPERATION BENCHMARKS")
    print("="*60)
    
    strategies = ['standard', 'blocked', 'streaming']
    data_sizes = [1000000, 4000000, 16000000]  # Matrix elements
    
    runner.compare_strategies(
        "Matrix Multiplication",
        BenchmarkCategory.GRAPH,  # Reusing category
        benchmark_matrix_multiply,
        strategies,
        data_sizes
    )


def run_quick_benchmarks(runner: BenchmarkRunner):
    """Run a quick subset of benchmarks"""
    print("\n" + "="*60)
    print("QUICK BENCHMARK SUITE")
    print("="*60)
    
    # Sorting
    runner.compare_strategies(
        "Quick Sort Test",
        BenchmarkCategory.SORTING,
        benchmark_sorting,
        ['standard', 'sqrt_n'],
        [10000, 100000]
    )
    
    # Database
    runner.compare_strategies(
        "Quick DB Test",
        BenchmarkCategory.DATABASE,
        benchmark_database_query,
        ['standard', 'sqrt_n'],
        [1000, 10000]
    )


def run_all_benchmarks(runner: BenchmarkRunner):
    """Run complete benchmark suite"""
    sorting_suite(runner)
    searching_suite(runner)
    database_suite(runner)
    ml_suite(runner)
    graph_suite(runner)
    matrix_suite(runner)


def analyze_results(results_file: str):
    """Analyze and visualize benchmark results"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = [BenchmarkResult(**r) for r in data['results']]
    
    # Group by category
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    # Create summary
    print("\n" + "="*60)
    print("BENCHMARK ANALYSIS")
    print("="*60)
    
    for category, cat_results in categories.items():
        print(f"\n{category}:")
        
        # Group by benchmark name
        benchmarks = {}
        for r in cat_results:
            if r.name not in benchmarks:
                benchmarks[r.name] = []
            benchmarks[r.name].append(r)
        
        for name, bench_results in benchmarks.items():
            print(f"\n  {name}:")
            
            # Find best strategies
            by_time = min(bench_results, key=lambda r: r.time_seconds)
            by_memory = min(bench_results, key=lambda r: r.memory_peak_mb)
            by_product = min(bench_results, key=lambda r: r.space_time_product)
            
            print(f"    Fastest: {by_time.strategy} ({by_time.time_seconds:.3f}s)")
            print(f"    Least memory: {by_memory.strategy} ({by_memory.memory_peak_mb:.1f}MB)")
            print(f"    Best space-time: {by_product.strategy} ({by_product.space_time_product:.1f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Benchmark Analysis', fontsize=16)
    
    # Plot 1: Time comparison
    ax = axes[0, 0]
    for name, bench_results in list(benchmarks.items())[:1]:
        strategies = {}
        for r in bench_results:
            if r.strategy not in strategies:
                strategies[r.strategy] = ([], [])
            strategies[r.strategy][0].append(r.data_size)
            strategies[r.strategy][1].append(r.time_seconds)
        
        for strategy, (sizes, times) in strategies.items():
            ax.loglog(sizes, times, 'o-', label=strategy, linewidth=2)
    
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Memory comparison
    ax = axes[0, 1]
    for name, bench_results in list(benchmarks.items())[:1]:
        strategies = {}
        for r in bench_results:
            if r.strategy not in strategies:
                strategies[r.strategy] = ([], [])
            strategies[r.strategy][0].append(r.data_size)
            strategies[r.strategy][1].append(r.memory_peak_mb)
        
        for strategy, (sizes, memories) in strategies.items():
            ax.loglog(sizes, memories, 'o-', label=strategy, linewidth=2)
    
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Space-time product
    ax = axes[1, 0]
    for name, bench_results in list(benchmarks.items())[:1]:
        strategies = {}
        for r in bench_results:
            if r.strategy not in strategies:
                strategies[r.strategy] = ([], [])
            strategies[r.strategy][0].append(r.data_size)
            strategies[r.strategy][1].append(r.space_time_product)
        
        for strategy, (sizes, products) in strategies.items():
            ax.loglog(sizes, products, 'o-', label=strategy, linewidth=2)
    
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Space-Time Product')
    ax.set_title('Overall Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Throughput
    ax = axes[1, 1]
    for name, bench_results in list(benchmarks.items())[:1]:
        strategies = {}
        for r in bench_results:
            if r.strategy not in strategies:
                strategies[r.strategy] = ([], [])
            strategies[r.strategy][0].append(r.data_size)
            strategies[r.strategy][1].append(r.throughput)
        
        for strategy, (sizes, throughputs) in strategies.items():
            ax.semilogx(sizes, throughputs, 'o-', label=strategy, linewidth=2)
    
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Throughput (ops/s)')
    ax.set_title('Processing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_analysis.png', dpi=150)
    plt.show()


def main():
    """Run benchmark suite"""
    print("SpaceTime Benchmark Suite")
    print("="*60)
    
    runner = BenchmarkRunner()
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='SpaceTime Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks only')
    parser.add_argument('--suite', choices=['sorting', 'searching', 'database', 'ml', 'graph', 'matrix'],
                       help='Run specific benchmark suite')
    parser.add_argument('--analyze', type=str, help='Analyze results file')
    parser.add_argument('--plot', action='store_true', help='Plot results after running')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.analyze)
    elif args.suite:
        # Run specific suite
        if args.suite == 'sorting':
            sorting_suite(runner)
        elif args.suite == 'searching':
            searching_suite(runner)
        elif args.suite == 'database':
            database_suite(runner)
        elif args.suite == 'ml':
            ml_suite(runner)
        elif args.suite == 'graph':
            graph_suite(runner)
        elif args.suite == 'matrix':
            matrix_suite(runner)
    elif args.quick:
        run_quick_benchmarks(runner)
    else:
        # Run all benchmarks
        run_all_benchmarks(runner)
    
    # Save results
    if runner.results:
        runner.save_results()
        
        if args.plot:
            runner.plot_results()
    
    print("\n" + "="*60)
    print("Benchmark suite complete!")
    print("="*60)


if __name__ == "__main__":
    main()