# SpaceTime Benchmark Suite

Standardized benchmarks for measuring and comparing space-time tradeoffs across algorithms and systems.

## Features

- **Standard Benchmarks**: Sorting, searching, graph algorithms, matrix operations
- **Real-World Workloads**: Database queries, ML training, distributed computing
- **Accurate Measurement**: Time, memory (peak/average), cache misses, throughput
- **Statistical Analysis**: Compare strategies with confidence
- **Reproducible Results**: Controlled environment, result validation
- **Visualization**: Automatic plots and analysis

## Installation

```bash
# From sqrtspace-tools root directory
pip install numpy matplotlib psutil

# For database benchmarks
pip install sqlite3  # Usually pre-installed
```

## Quick Start

```bash
# Run quick benchmark suite
python spacetime_benchmarks.py --quick

# Run all benchmarks
python spacetime_benchmarks.py

# Run specific suite
python spacetime_benchmarks.py --suite sorting

# Analyze saved results
python spacetime_benchmarks.py --analyze results_20240315_143022.json
```

## Benchmark Categories

### 1. Sorting Algorithms
Compare memory-time tradeoffs in sorting:

```python
# Strategies benchmarked:
- standard: In-memory quicksort/mergesort (O(n) space)
- sqrt_n: External sort with √n buffer (O(√n) space)
- constant: Streaming sort (O(1) space)

# Example results for n=1,000,000:
Standard: 0.125s, 8.0MB memory
√n buffer: 0.187s, 0.3MB memory (96% less memory, 50% slower)
Streaming: 0.543s, 0.01MB memory (99.9% less memory, 4.3x slower)
```

### 2. Search Data Structures
Compare different index structures:

```python
# Strategies benchmarked:
- hash: Standard hash table (O(n) space)
- btree: B-tree index (O(n) space, cache-friendly)
- external: External index with √n cache

# Example results for n=1,000,000:
Hash table: 0.003s per query, 40MB memory
B-tree: 0.008s per query, 35MB memory
External: 0.025s per query, 2MB memory (95% less)
```

### 3. Database Operations
Real SQLite database with different cache configurations:

```python
# Strategies benchmarked:
- standard: Default cache size (2000 pages)
- sqrt_n: √n cache pages
- minimal: Minimal cache (10 pages)

# Example results for n=100,000 rows:
Standard: 1000 queries in 0.45s, 16MB cache
√n cache: 1000 queries in 0.52s, 1.2MB cache
Minimal: 1000 queries in 1.83s, 0.08MB cache
```

### 4. ML Training
Neural network training with memory optimizations:

```python
# Strategies benchmarked:
- standard: Keep all activations for backprop
- gradient_checkpoint: Recompute activations (√n checkpoints)
- mixed_precision: FP16 compute, FP32 master weights

# Example results for 50,000 samples:
Standard: 2.3s, 195MB peak memory
Checkpointing: 2.8s, 42MB peak memory (78% less)
Mixed precision: 2.1s, 98MB peak memory (50% less)
```

### 5. Graph Algorithms
Graph traversal with memory constraints:

```python
# Strategies benchmarked:
- bfs: Standard breadth-first search
- dfs_iterative: Depth-first with explicit stack
- memory_bounded: Limited queue size (like IDA*)

# Example results for n=50,000 nodes:
BFS: 0.18s, 12MB memory (full frontier)
DFS: 0.15s, 4MB memory (stack only)
Bounded: 0.31s, 0.8MB memory (√n queue)
```

### 6. Matrix Operations
Cache-aware matrix multiplication:

```python
# Strategies benchmarked:
- standard: Naive multiplication
- blocked: Cache-blocked multiplication
- streaming: Row-by-row streaming

# Example results for 2000×2000 matrices:
Standard: 1.2s, 32MB memory
Blocked: 0.8s, 32MB memory (33% faster)
Streaming: 3.5s, 0.5MB memory (98% less memory)
```

## Running Benchmarks

### Command Line Options

```bash
# Run all benchmarks
python spacetime_benchmarks.py

# Quick benchmarks (subset for testing)
python spacetime_benchmarks.py --quick

# Specific suite only
python spacetime_benchmarks.py --suite sorting
python spacetime_benchmarks.py --suite database
python spacetime_benchmarks.py --suite ml

# With automatic plotting
python spacetime_benchmarks.py --plot

# Analyze previous results
python spacetime_benchmarks.py --analyze results_20240315_143022.json
```

### Programmatic Usage

```python
from spacetime_benchmarks import BenchmarkRunner, benchmark_sorting

runner = BenchmarkRunner()

# Run single benchmark
result = runner.run_benchmark(
    name="Custom Sort",
    category=BenchmarkCategory.SORTING,
    strategy="sqrt_n",
    benchmark_func=benchmark_sorting,
    data_size=1000000
)

print(f"Time: {result.time_seconds:.3f}s")
print(f"Memory: {result.memory_peak_mb:.1f}MB")
print(f"Space-Time Product: {result.space_time_product:.1f}")

# Compare strategies
comparisons = runner.compare_strategies(
    name="Sort Comparison",
    category=BenchmarkCategory.SORTING,
    benchmark_func=benchmark_sorting,
    strategies=["standard", "sqrt_n", "constant"],
    data_sizes=[10000, 100000, 1000000]
)

for comp in comparisons:
    print(f"\n{comp.baseline.strategy} vs {comp.optimized.strategy}:")
    print(f"  Memory reduction: {comp.memory_reduction:.1f}%")
    print(f"  Time overhead: {comp.time_overhead:.1f}%")
    print(f"  Recommendation: {comp.recommendation}")
```

## Custom Benchmarks

Add your own benchmarks:

```python
def benchmark_custom_algorithm(n: int, strategy: str = 'standard', **kwargs) -> int:
    """Custom algorithm with space-time tradeoffs"""
    
    if strategy == 'standard':
        # O(n) space implementation
        data = list(range(n))
        # ... algorithm ...
        return n  # Return operation count
        
    elif strategy == 'memory_efficient':
        # O(√n) space implementation
        buffer_size = int(np.sqrt(n))
        # ... algorithm ...
        return n
        
# Register and run
runner = BenchmarkRunner()
runner.compare_strategies(
    "Custom Algorithm",
    BenchmarkCategory.CUSTOM,
    benchmark_custom_algorithm,
    ["standard", "memory_efficient"],
    [1000, 10000, 100000]
)
```

## Understanding Results

### Key Metrics

1. **Time (seconds)**: Wall-clock execution time
2. **Peak Memory (MB)**: Maximum memory usage during execution
3. **Average Memory (MB)**: Average memory over execution
4. **Throughput (ops/sec)**: Operations completed per second
5. **Space-Time Product**: Memory × Time (lower is better)

### Interpreting Comparisons

```
Comparison standard vs sqrt_n:
  Memory reduction: 94.3%      # How much less memory
  Time overhead: 47.2%         # How much slower
  Space-time improvement: 91.8% # Overall efficiency gain
  Recommendation: Use sqrt_n for 94% memory savings
```

### When to Use Each Strategy

| Strategy | Use When | Avoid When |
|----------|----------|------------|
| Standard | Memory abundant, Speed critical | Memory constrained |
| √n Optimized | Memory limited, Moderate slowdown OK | Real-time systems |
| O(log n) | Extreme memory constraints | Random access needed |
| O(1) Space | Streaming data, Minimal memory | Need multiple passes |

## Benchmark Output

### Results File Format

```json
{
  "system_info": {
    "cpu_count": 8,
    "memory_gb": 32.0,
    "l3_cache_mb": 12.0
  },
  "results": [
    {
      "name": "Sorting",
      "category": "sorting",
      "strategy": "sqrt_n",
      "data_size": 1000000,
      "time_seconds": 0.187,
      "memory_peak_mb": 8.2,
      "memory_avg_mb": 6.5,
      "throughput": 5347593.5,
      "space_time_product": 1.534,
      "metadata": {
        "success": true,
        "operations": 1000000
      }
    }
  ],
  "timestamp": 1710512345.678
}
```

### Visualization

Automatic plots show:
- Time complexity curves
- Memory usage scaling
- Space-time product comparison
- Throughput vs data size

## Performance Tips

1. **System Preparation**:
   ```bash
   # Disable CPU frequency scaling
   sudo cpupower frequency-set -g performance
   
   # Clear caches
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

2. **Accurate Memory Measurement**:
   - Results include Python overhead
   - Use `memory_peak_mb` for maximum usage
   - `memory_avg_mb` shows typical usage

3. **Reproducibility**:
   - Run multiple times and average
   - Control background processes
   - Use consistent data sizes

## Extending the Suite

### Adding New Categories

```python
class BenchmarkCategory(Enum):
    # ... existing categories ...
    CUSTOM = "custom"

def custom_suite(runner: BenchmarkRunner):
    """Run custom benchmarks"""
    strategies = ['approach1', 'approach2']
    data_sizes = [1000, 10000, 100000]
    
    runner.compare_strategies(
        "Custom Workload",
        BenchmarkCategory.CUSTOM,
        benchmark_custom,
        strategies,
        data_sizes
    )
```

### Platform-Specific Metrics

```python
def get_cache_misses():
    """Get L3 cache misses (Linux perf)"""
    if platform.system() == 'Linux':
        # Use perf_event_open or read from perf
        pass
    return None
```

## Real-World Insights

From our benchmarks:

1. **√n strategies typically save 90-99% memory** with 20-100% time overhead

2. **Cache-aware algorithms can be faster** despite theoretical complexity

3. **Memory bandwidth often dominates** over computational complexity

4. **Optimal strategy depends on**:
   - Data size vs available memory
   - Latency requirements
   - Power/cost constraints

## Troubleshooting

### Memory Measurements Seem Low
- Python may not release memory immediately
- Use `gc.collect()` before benchmarks
- Check for lazy evaluation

### High Variance in Results
- Disable CPU throttling
- Close other applications  
- Increase data sizes for stability

### Database Benchmarks Fail
- Ensure write permissions in output directory
- Check SQLite installation
- Verify disk space available

## Contributing

Add new benchmarks following the pattern:

1. Implement `benchmark_*` function
2. Return operation count
3. Handle different strategies
4. Add suite function
5. Update documentation

## See Also

- [SpaceTimeCore](../core/spacetime_core.py): Core calculations
- [Profiler](../profiler/): Profile your applications
- [Visual Explorer](../explorer/): Visualize tradeoffs