# SpaceTime Compiler Plugin

Compile-time optimization tool that automatically identifies and applies space-time tradeoffs in Python code.

## Features

- **AST Analysis**: Parse and analyze Python code for optimization opportunities
- **Automatic Transformation**: Convert algorithms to use √n memory strategies
- **Safety Preservation**: Ensure correctness while optimizing
- **Static Memory Analysis**: Predict memory usage before runtime
- **Code Generation**: Produce readable, optimized Python code
- **Detailed Reports**: Understand what optimizations were applied and why

## Installation

```bash
# From sqrtspace-tools root directory
pip install ast numpy
```

## Quick Start

### Command Line Usage

```bash
# Analyze code for opportunities
python spacetime_compiler.py my_code.py --analyze-only

# Compile with optimizations
python spacetime_compiler.py my_code.py -o optimized_code.py

# Generate optimization report
python spacetime_compiler.py my_code.py -o optimized.py -r report.txt

# Run demonstration
python spacetime_compiler.py --demo
```

### Programmatic Usage

```python
from spacetime_compiler import SpaceTimeCompiler

compiler = SpaceTimeCompiler()

# Analyze a file
opportunities = compiler.analyze_file('my_algorithm.py')
for opp in opportunities:
    print(f"Line {opp.line_number}: {opp.description}")
    print(f"  Memory savings: {opp.memory_savings}%")

# Transform code
with open('my_algorithm.py', 'r') as f:
    code = f.read()

result = compiler.transform_code(code)
print(f"Memory reduction: {result.estimated_memory_reduction}%")
print(f"Optimized code:\n{result.optimized_code}")
```

### Decorator Usage

```python
from spacetime_compiler import optimize_spacetime

@optimize_spacetime()
def process_large_dataset(data):
    # Original code
    results = []
    for item in data:
        processed = expensive_operation(item)
        results.append(processed)
    return results

# Function is automatically optimized at definition time
# Will use √n checkpointing and streaming where beneficial
```

## Optimization Types

### 1. Checkpoint Insertion
Identifies loops with accumulation and adds √n checkpointing:

```python
# Before
total = 0
for i in range(1000000):
    total += expensive_computation(i)

# After
total = 0
sqrt_n = int(np.sqrt(1000000))
checkpoint_total = 0
for i in range(1000000):
    total += expensive_computation(i)
    if i % sqrt_n == 0:
        checkpoint_total = total  # Checkpoint
```

### 2. Buffer Size Optimization
Converts fixed buffers to √n sizing:

```python
# Before
buffer = []
for item in huge_dataset:
    buffer.append(process(item))
    if len(buffer) >= 10000:
        flush_buffer(buffer)
        buffer = []

# After
buffer_size = int(np.sqrt(len(huge_dataset)))
buffer = []
for item in huge_dataset:
    buffer.append(process(item))
    if len(buffer) >= buffer_size:
        flush_buffer(buffer)
        buffer = []
```

### 3. Streaming Conversion
Converts list comprehensions to generators:

```python
# Before
squares = [x**2 for x in range(1000000)]  # 8MB memory

# After  
squares = (x**2 for x in range(1000000))  # ~0 memory
```

### 4. External Memory Algorithms
Replaces in-memory operations with external variants:

```python
# Before
sorted_data = sorted(huge_list)

# After
sorted_data = external_sort(huge_list, 
                           buffer_size=int(np.sqrt(len(huge_list))))
```

### 5. Cache Blocking
Optimizes matrix and array operations:

```python
# Before
C = np.dot(A, B)  # Cache thrashing for large matrices

# After
C = blocked_matmul(A, B, block_size=64)  # Cache-friendly
```

## How It Works

### 1. AST Analysis Phase
```python
# The compiler parses code into Abstract Syntax Tree
tree = ast.parse(source_code)

# Custom visitor identifies patterns
analyzer = SpaceTimeAnalyzer()
analyzer.visit(tree)

# Returns list of opportunities with metadata
opportunities = analyzer.opportunities
```

### 2. Transformation Phase
```python
# Transformer modifies AST nodes
transformer = SpaceTimeTransformer(opportunities)
optimized_tree = transformer.visit(tree)

# Generate Python code from modified AST
optimized_code = ast.unparse(optimized_tree)
```

### 3. Code Generation
- Adds necessary imports
- Preserves code structure and readability
- Includes comments explaining optimizations
- Maintains compatibility

## Optimization Criteria

The compiler uses these criteria to decide on optimizations:

| Criterion | Weight | Description |
|-----------|---------|-------------|
| Memory Savings | 40% | Estimated memory reduction |
| Time Overhead | 30% | Performance impact |
| Confidence | 20% | Certainty of analysis |
| Code Clarity | 10% | Readability preservation |

### Automatic Selection Logic
```python
def should_apply(opportunity):
    if opportunity.confidence < 0.7:
        return False  # Too uncertain
    
    if opportunity.memory_savings > 50 and opportunity.time_overhead < 100:
        return True  # Good tradeoff
    
    if opportunity.time_overhead < 0:
        return True  # Performance improvement!
    
    return False
```

## Example Transformations

### Example 1: Data Processing Pipeline
```python
# Original code
def process_logs(log_files):
    all_entries = []
    for file in log_files:
        entries = parse_file(file)
        all_entries.extend(entries)
    
    sorted_entries = sorted(all_entries, key=lambda x: x.timestamp)
    
    aggregated = {}
    for entry in sorted_entries:
        key = entry.user_id
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(entry)
    
    return aggregated

# Compiler identifies:
# - Large accumulation in all_entries
# - Sorting operation on potentially large data
# - Dictionary building with lists

# Optimized code
def process_logs(log_files):
    # Use generator to avoid storing all entries
    def entry_generator():
        for file in log_files:
            entries = parse_file(file)
            yield from entries
    
    # External sort with √n memory
    sorted_entries = external_sort(
        entry_generator(), 
        key=lambda x: x.timestamp,
        buffer_size=int(np.sqrt(estimate_total_entries()))
    )
    
    # Streaming aggregation
    aggregated = {}
    for entry in sorted_entries:
        key = entry.user_id
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(entry)
        
        # Checkpoint large user lists
        if len(aggregated[key]) % int(np.sqrt(len(aggregated[key]))) == 0:
            checkpoint_user_data(key, aggregated[key])
    
    return aggregated
```

### Example 2: Scientific Computing
```python
# Original code
def simulate_particles(n_steps, n_particles):
    positions = np.random.rand(n_particles, 3)
    velocities = np.random.rand(n_particles, 3)
    forces = np.zeros((n_particles, 3))
    
    trajectory = []
    
    for step in range(n_steps):
        # Calculate forces between all pairs
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                force = calculate_force(positions[i], positions[j])
                forces[i] += force
                forces[j] -= force
        
        # Update positions
        positions += velocities * dt
        velocities += forces * dt / mass
        
        # Store trajectory
        trajectory.append(positions.copy())
    
    return trajectory

# Optimized code
def simulate_particles(n_steps, n_particles):
    positions = np.random.rand(n_particles, 3)
    velocities = np.random.rand(n_particles, 3)
    forces = np.zeros((n_particles, 3))
    
    # √n checkpointing for trajectory
    checkpoint_interval = int(np.sqrt(n_steps))
    trajectory_checkpoints = []
    current_trajectory = []
    
    # Blocked force calculation for cache efficiency
    block_size = min(64, int(np.sqrt(n_particles)))
    
    for step in range(n_steps):
        # Blocked force calculation
        for i_block in range(0, n_particles, block_size):
            for j_block in range(i_block, n_particles, block_size):
                # Process block
                for i in range(i_block, min(i_block + block_size, n_particles)):
                    for j in range(max(i+1, j_block), 
                                 min(j_block + block_size, n_particles)):
                        force = calculate_force(positions[i], positions[j])
                        forces[i] += force
                        forces[j] -= force
        
        # Update positions
        positions += velocities * dt
        velocities += forces * dt / mass
        
        # Checkpoint trajectory
        current_trajectory.append(positions.copy())
        if step % checkpoint_interval == 0:
            trajectory_checkpoints.append(current_trajectory)
            current_trajectory = []
    
    # Reconstruct full trajectory on demand
    return CheckpointedTrajectory(trajectory_checkpoints, current_trajectory)
```

## Report Format

The compiler generates detailed reports:

```
SpaceTime Compiler Optimization Report
============================================================

Opportunities found: 5
Optimizations applied: 3
Estimated memory reduction: 87.3%
Estimated time overhead: 23.5%

Optimization Opportunities Found:
------------------------------------------------------------
1. [✓] Line 145: checkpoint
   Large loop with accumulation - consider √n checkpointing
   Memory savings: 95.0%
   Time overhead: 20.0%
   Confidence: 0.85

2. [✓] Line 203: external_memory
   Sorting large data - consider external sort with √n memory
   Memory savings: 93.0%
   Time overhead: 45.0%
   Confidence: 0.72

3. [✗] Line 67: streaming
   Large list comprehension - consider generator expression
   Memory savings: 99.0%
   Time overhead: 5.0%
   Confidence: 0.65  (Not applied: confidence too low)

4. [✓] Line 234: cache_blocking
   Matrix operation - consider cache-blocked implementation
   Memory savings: 0.0%
   Time overhead: -30.0%  (Performance improvement!)
   Confidence: 0.88

5. [✗] Line 89: buffer_size
   Buffer operations in loop - consider √n buffer sizing
   Memory savings: 90.0%
   Time overhead: 15.0%
   Confidence: 0.60  (Not applied: confidence too low)
```

## Integration with Build Systems

### setup.py Integration
```python
from setuptools import setup
from spacetime_compiler import compile_package

setup(
    name='my_package',
    cmdclass={
        'build_py': compile_package,  # Auto-optimize during build
    }
)
```

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: spacetime-optimize
        name: SpaceTime Optimization
        entry: python -m spacetime_compiler
        language: system
        files: \.py$
        args: [--analyze-only]
```

## Safety and Correctness

The compiler ensures safety through:

1. **Conservative Transformation**: Only applies high-confidence optimizations
2. **Semantic Preservation**: Maintains exact program behavior
3. **Type Safety**: Preserves type signatures and contracts
4. **Error Handling**: Maintains exception behavior
5. **Testing**: Recommends testing optimized code

## Limitations

1. **Python Only**: Currently supports Python AST only
2. **Static Analysis**: Cannot optimize runtime-dependent patterns
3. **Import Dependencies**: Optimized code may require additional imports
4. **Readability**: Some optimizations may reduce code clarity
5. **Not All Patterns**: Limited to recognized optimization patterns

## Future Enhancements

- Support for more languages (C++, Java, Rust)
- Integration with IDEs (VS Code, PyCharm)
- Profile-guided optimization
- Machine learning for pattern recognition
- Automatic benchmark generation
- Distributed system optimizations

## Troubleshooting

### "Optimization not applied"
- Check confidence thresholds
- Ensure pattern matches expected structure
- Verify data size estimates

### "Import errors in optimized code"
- Install required dependencies (external_sort, etc.)
- Check import statements in generated code

### "Different behavior after optimization"
- File a bug report with minimal example
- Use --analyze-only to review planned changes
- Test with smaller datasets first

## Contributing

To add new optimization patterns:

1. Add pattern detection in `SpaceTimeAnalyzer`
2. Implement transformation in `SpaceTimeTransformer`
3. Add tests for correctness
4. Update documentation

## See Also

- [SpaceTimeCore](../core/spacetime_core.py): Core calculations
- [Profiler](../profiler/): Runtime profiling
- [Benchmarks](../benchmarks/): Performance testing