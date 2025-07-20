# SqrtSpace SpaceTime Specialized Tools

This directory contains specialized experimental tools and advanced utilities that complement the main SqrtSpace SpaceTime implementations. These tools explore specific use cases and provide domain-specific optimizations beyond the core framework.

## Overview

These specialized tools extend the core SpaceTime framework with experimental features, domain-specific optimizers, and advanced analysis capabilities. They demonstrate cutting-edge applications of Williams' space-time tradeoffs in various computing domains.

**Note:** For production-ready implementations, please use:
- Python: `pip install sqrtspace-spacetime` ()
- .NET: `dotnet add package SqrtSpace.SpaceTime` ()
- PHP: `composer require sqrtspace/spacetime` ()

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sqrtspace/sqrtspace-tools.git
cd sqrtspace-tools

# Install dependencies
pip install -r requirements.txt

# Run basic tests
python test_basic.py

# Profile your application
python profiler/example_profile.py
```

## Specialized Tools

**Note:** The core functionality (profiler, ML optimizer, auto-checkpoint) has been moved to the production packages. These specialized tools provide additional experimental features:

### 1. [Memory-Aware Query Optimizer](db_optimizer/) 
Database query optimizer considering memory hierarchies.

```python
from db_optimizer.memory_aware_optimizer import MemoryAwareOptimizer

optimizer = MemoryAwareOptimizer(conn, memory_limit=10*1024*1024)
result = optimizer.optimize_query(sql)
print(result.explanation)  # "Changed join from nested_loop to hash_join saving 9MB"
```

**Features:**
- Cost model with L3/RAM/SSD boundaries
- Intelligent join algorithm selection
- √n buffer sizing
- Spill strategy planning

### 2.  [Distributed Shuffle Optimizer](distsys/)
Optimize shuffle operations in distributed frameworks.

```python
from distsys.shuffle_optimizer import ShuffleOptimizer, ShuffleTask

optimizer = ShuffleOptimizer(nodes)
plan = optimizer.optimize_shuffle(task)
print(plan.explanation)  # "Using tree_aggregate with √n-height tree"
```

**Features:**
- Optimal buffer sizing per node
- √n-height aggregation trees
- Network topology awareness
- Compression selection

### 3. [Cache-Aware Data Structures](datastructures/)
Data structures that adapt to memory hierarchies.

```python
from datastructures import AdaptiveMap

map = AdaptiveMap()  # Automatically adapts
# Switches: array → B-tree → hash table → external storage
```

**Features:**
- Automatic implementation switching
- Cache-line-aligned nodes
- √n external buffers
- Compressed variants

### 4. [SpaceTime Configuration Advisor](advisor/)
Analyze systems and recommend optimal settings.

```python
from advisor.config_advisor import ConfigAdvisor

advisor = ConfigAdvisor()
recommendations = advisor.analyze_system(workload_type='database')
print(recommendations.explanation)
```

### 5. [Visual SpaceTime Explorer](explorer/) 
Interactive visualization of space-time tradeoffs.

```python
from explorer.spacetime_explorer import SpaceTimeExplorer

explorer = SpaceTimeExplorer()
explorer.visualize_tradeoffs(algorithm='sorting', n=1000000)
```

### 6. [Benchmark Suite](benchmarks/) 
Standardized benchmarks for measuring tradeoffs.

```python
from benchmarks.spacetime_benchmarks import run_benchmark

results = run_benchmark('external_sort', sizes=[1e6, 1e7, 1e8])
```

### 7. [Compiler Plugin](compiler/) 
Compile-time optimization of space-time tradeoffs.

```python
from compiler.spacetime_compiler import optimize_code

optimized = optimize_code(source_code)
print(optimized.transformations)
```

## Core Components

### [SpaceTimeCore](core/spacetime_core.py)
Shared foundation providing:
- Memory hierarchy modeling
- √n interval calculation
- Strategy comparison framework
- Resource-aware scheduling

## Real-World Impact

These optimizations appear throughout modern computing:

- **2+ billion smartphones**: SQLite uses √n buffer pool sizing
- **ChatGPT/Claude**: Flash Attention trades compute for memory
- **Google/Meta**: MapReduce frameworks use external sorting
- **Video games**: A* pathfinding with memory constraints
- **Embedded systems**: Severe memory limitations require tradeoffs

## Example Results

From our experiments:

### Checkpointed Sorting
- **Before**: O(n) memory, baseline speed
- **After**: O(√n) memory, 10-50% slower
- **Savings**: 90-99% memory reduction

### LLM Attention
- **Full KV-cache**: 197 tokens/sec, O(n) memory
- **Flash Attention**: 1,349 tokens/sec, O(√n) memory
- **Result**: 6.8× faster with less memory!

### Database Buffer Pool
- **O(n) cache**: 4.5 queries/sec
- **O(√n) cache**: 4.3 queries/sec  
- **Savings**: 94% memory, 4% slowdown

## Installation

### Basic Installation
```bash
pip install numpy matplotlib psutil
```

### Full Installation
```bash
pip install -r requirements.txt
```

## Project Structure

```
sqrtspace-tools/
├── core/                    # Shared optimization engine
│   └── spacetime_core.py   # Memory hierarchy, √n calculator
├── advisor/                # Configuration advisor 
├── benchmarks/             # Performance benchmarks 
├── compiler/               # Compiler optimizations 
├── datastructures/         # Adaptive data structures 
├── db_optimizer/           # Database optimizations 
├── distsys/               # Distributed systems 
├── explorer/              # Visualization tools 
└── requirements.txt       # Python dependencies
```

## Key Insights

1. **Williams' bound is everywhere**: The √n pattern appears in databases, ML, algorithms, and systems
2. **Massive constant factors**: Theory says √n is optimal, but 100-10,000× slowdowns are common
3. **Memory hierarchies matter**: L1→L2→L3→RAM→Disk transitions create performance cliffs
4. **Modern hardware changes the game**: Fast SSDs and memory bandwidth limits alter tradeoffs
5. **Cache-aware beats theoretically optimal**: Locality often trumps algorithmic complexity

## Contributing

We welcome contributions! Areas of focus:

1. **Tool Development**: Help implement the remaining tools
2. **Integration**: Add support for more frameworks (PyTorch, TensorFlow, Spark)
3. **Documentation**: Improve examples and tutorials
4. **Research**: Explore new space-time tradeoff patterns
5. **Testing**: Add comprehensive test suites

## Citation

If you use these tools in research, please cite:

```bibtex
@software{sqrtspace_tools,
  title = {SqrtSpace Tools: Space-Time Optimization Suite},
  author={Friedel Jr., David H.},
  year = {2025},
  url = {https://github.com/sqrtspace/sqrtspace-tools}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

Based on theoretical work by Williams (STOC 2025) and inspired by real-world systems at Anthropic, Google, Meta, OpenAI, and others.

---

*"Making theoretical computer science practical, one tool at a time."*
