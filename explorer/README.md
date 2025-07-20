# Visual SpaceTime Explorer

Interactive visualization tool for understanding and exploring space-time tradeoffs in algorithms and systems.

## Features

- **Interactive Plots**: Pan, zoom, and explore tradeoff curves in real-time
- **Live Parameter Updates**: See immediate impact of changing data sizes and strategies
- **Multiple Visualizations**: Memory hierarchy, checkpoint intervals, cost analysis, 3D views
- **Educational Mode**: Learn theoretical concepts through visual demonstrations
- **Export Capabilities**: Save analyses and plots for presentations or reports

## Installation

```bash
# From sqrtspace-tools root directory
pip install matplotlib numpy

# For full features including animations
pip install matplotlib numpy scipy
```

## Quick Start

```python
from explorer import SpaceTimeVisualizer

# Launch interactive explorer
visualizer = SpaceTimeVisualizer()
visualizer.create_main_window()

# The explorer will open with:
# - Main tradeoff curves
# - Memory hierarchy view
# - Checkpoint visualization
# - Cost analysis
# - Performance metrics
# - 3D space-time-cost plot
```

## Interactive Controls

### Sliders
- **Data Size**: Adjust n from 100 to 1 billion (log scale)
- See how different algorithms scale with data size

### Radio Buttons
- **Strategy**: Choose between sqrt_n, linear, log_n, constant
- **View**: Switch between tradeoff, animated, comparison views

### Mouse Controls
- **Pan**: Click and drag on plots
- **Zoom**: Scroll wheel or right-click drag
- **Reset**: Double-click to reset view

### Export Button
- Save current analysis as JSON
- Export plots as high-resolution PNG

## Visualization Types

### 1. Main Tradeoff Curves
Shows theoretical and practical space-time tradeoffs:

```python
# The main plot displays:
- O(n) space algorithms (standard)
- O(√n) space algorithms (Williams' bound)
- O(log n) space algorithms (compressed)
- O(1) space algorithms (streaming)
- Feasible region (gray shaded area)
- Current configuration (red dot)
```

### 2. Memory Hierarchy View
Visualizes data distribution across cache levels:

```python
# Shows how data is placed in:
- L1 Cache (32KB, 1ns)
- L2 Cache (256KB, 3ns)
- L3 Cache (8MB, 12ns)
- RAM (32GB, 100ns)
- SSD (512GB, 10μs)
```

### 3. Checkpoint Intervals
Compares different checkpointing strategies:

```python
# Strategies visualized:
- No checkpointing (full memory)
- √n intervals (optimal)
- Fixed intervals (e.g., every 1000)
- Exponential intervals (doubling)
```

### 4. Cost Analysis
Breaks down costs by component:

```python
# Cost factors:
- Memory cost (cloud storage)
- Time cost (compute hours)
- Total cost (combined)
- Comparison across strategies
```

### 5. Performance Metrics
Radar chart showing multiple dimensions:

```python
# Metrics evaluated:
- Memory Efficiency (0-100%)
- Speed (0-100%)
- Fault Tolerance (0-100%)
- Scalability (0-100%)
- Cost Efficiency (0-100%)
```

### 6. 3D Visualization
Three-dimensional view of space-time-cost:

```python
# Axes:
- X: log₁₀(Space)
- Y: log₁₀(Time)
- Z: log₁₀(Cost)
# Shows tradeoff surfaces for different strategies
```

## Example Visualizations

Run comprehensive examples:

```bash
python example_visualizations.py
```

This creates four sets of visualizations:

### 1. Algorithm Comparison
- Sorting algorithms (QuickSort vs MergeSort vs External Sort)
- Search structures (Array vs BST vs Hash vs B-tree)
- Matrix multiplication strategies
- Graph algorithms with memory constraints

### 2. Real-World Systems
- Database buffer pool strategies
- LLM inference with KV-cache optimization
- MapReduce shuffle strategies
- Mobile app memory management

### 3. Optimization Impact
- Memory reduction factors (10x to 1,000,000x)
- Time overhead analysis
- Cloud cost analysis
- Breakeven calculations

### 4. Educational Diagrams
- Williams' space-time bound
- Memory hierarchy and latencies
- Checkpoint strategy comparison
- Cache line utilization
- Algorithm selection guide
- Cost-benefit spider charts

## Use Cases

### 1. Algorithm Design
```python
# Compare different algorithm implementations
visualizer.current_n = 10**6  # 1 million elements
visualizer.update_all_plots()

# See which strategy is optimal for your data size
```

### 2. System Tuning
```python
# Analyze memory hierarchy impact
# Adjust parameters to match your system
hierarchy = MemoryHierarchy.detect_system()
visualizer.hierarchy = hierarchy
```

### 3. Education
```python
# Create educational visualizations
from example_visualizations import create_educational_diagrams
create_educational_diagrams()

# Perfect for teaching space-time tradeoffs
```

### 4. Research
```python
# Export data for analysis
visualizer._export_data(None)

# Creates JSON with all metrics and parameters
# Saves high-resolution plots
```

## Advanced Features

### Custom Strategies
Add your own algorithms:

```python
class CustomVisualizer(SpaceTimeVisualizer):
    def _get_strategy_metrics(self, n, strategy):
        if strategy == 'my_algorithm':
            space = n ** 0.7  # Custom space complexity
            time = n * np.log(n) ** 2  # Custom time
            cost = space * 0.1 + time * 0.01
            return space, time, cost
        return super()._get_strategy_metrics(n, strategy)
```

### Animation Mode
View algorithms in action:

```python
# Launch animated view
visualizer.create_animated_view()

# Shows:
# - Processing progress
# - Checkpoint creation
# - Memory usage over time
```

### Comparison Mode
Side-by-side strategy comparison:

```python
# Launch comparison view
visualizer.create_comparison_view()

# Creates 2x2 grid comparing all strategies
```

## Understanding the Visualizations

### Space-Time Curves
- **Lower-left**: Better (less space, less time)
- **Upper-right**: Worse (more space, more time)
- **Gray region**: Theoretically impossible
- **Green region**: Feasible implementations

### Memory Distribution
- **Darker colors**: Faster memory (L1, L2)
- **Lighter colors**: Slower memory (RAM, SSD)
- **Bar width**: Amount of data in that level
- **Numbers**: Access latency in nanoseconds

### Checkpoint Timeline
- **Blocks**: Work between checkpoints
- **Width**: Amount of progress
- **Gaps**: Checkpoint operations
- **Colors**: Different strategies

### Cost Analysis
- **Log scale**: Costs vary by orders of magnitude
- **Red outline**: Currently selected strategy
- **Bar height**: Relative cost (lower is better)

## Tips for Best Results

1. **Start with your actual data size**: Use the slider to match your workload

2. **Consider all metrics**: Don't optimize for memory alone - check time and cost

3. **Test edge cases**: Try very small and very large data sizes

4. **Export findings**: Save configurations that work well

5. **Compare strategies**: Use the comparison view for thorough analysis

## Interpreting Results

### When to use O(√n) strategies:
- Data size >> available memory
- Memory is expensive (cloud/embedded)
- Can tolerate 10-50% time overhead
- Need fault tolerance

### When to avoid:
- Data fits in memory
- Latency critical (< 10ms)
- Simple algorithms sufficient
- Overhead not justified

## Future Enhancements

- Real-time profiling integration
- Custom algorithm import
- Collaborative sharing
- AR/VR visualization
- Machine learning predictions

## See Also

- [SpaceTimeCore](../core/spacetime_core.py): Core calculations
- [Profiler](../profiler/): Profile your applications