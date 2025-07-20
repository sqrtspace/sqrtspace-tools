# SpaceTime Configuration Advisor

Intelligent system configuration advisor that applies Williams' √n space-time tradeoffs to optimize database, JVM, kernel, container, and application settings.

## Features

- **System Analysis**: Comprehensive hardware profiling (CPU, memory, storage, network)
- **Workload Characterization**: Analyze access patterns and resource requirements
- **Multi-System Support**: Database, JVM, kernel, container, and application configs
- **√n Optimization**: Apply theoretical bounds to real-world settings
- **A/B Testing**: Compare configurations with statistical confidence
- **AI Explanations**: Clear reasoning for each recommendation

## Installation

```bash
# From sqrtspace-tools root directory
pip install -r requirements-minimal.txt
```

## Quick Start

```python
from advisor import ConfigurationAdvisor, SystemType

advisor = ConfigurationAdvisor()

# Analyze for database workload
config = advisor.analyze(
    workload_data={
        'read_ratio': 0.8,
        'working_set_gb': 50,
        'total_data_gb': 500,
        'qps': 10000
    },
    target=SystemType.DATABASE
)

print(config.explanation)
# "Database configured with 12.5GB buffer pool (√n sizing), 
#  128MB work memory per operation, and standard checkpointing."
```

## System Types

### 1. Database Configuration
Optimizes PostgreSQL/MySQL settings:

```python
# E-commerce OLTP workload
config = advisor.analyze(
    workload_data={
        'read_ratio': 0.9,
        'working_set_gb': 20,
        'total_data_gb': 200,
        'qps': 5000,
        'connections': 300,
        'latency_sla_ms': 50
    },
    target=SystemType.DATABASE
)

# Generated PostgreSQL config:
# shared_buffers = 5120MB      # √n sized if data > memory
# work_mem = 21MB              # Per-operation memory
# checkpoint_segments = 16      # Based on write ratio
# max_connections = 600        # 2x concurrent users
```

### 2. JVM Configuration
Tunes heap size, GC, and thread settings:

```python
# Low-latency trading system
config = advisor.analyze(
    workload_data={
        'latency_sla_ms': 10,
        'working_set_gb': 8,
        'connections': 100
    },
    target=SystemType.JVM
)

# Generated JVM flags:
# -Xmx16g -Xms16g              # 50% of system memory
# -Xmn512m                     # √n young generation
# -XX:+UseG1GC                # Low-latency GC
# -XX:MaxGCPauseMillis=10     # Match SLA
```

### 3. Kernel Configuration
Optimizes Linux kernel parameters:

```python
# High-throughput web server
config = advisor.analyze(
    workload_data={
        'request_rate': 50000,
        'connections': 10000,
        'working_set_gb': 32
    },
    target=SystemType.KERNEL
)

# Generated sysctl settings:
# vm.dirty_ratio = 20
# vm.swappiness = 60
# net.core.somaxconn = 65535
# net.ipv4.tcp_max_syn_backlog = 65535
```

### 4. Container Configuration
Sets Docker/Kubernetes resource limits:

```python
# Microservice API
config = advisor.analyze(
    workload_data={
        'working_set_gb': 2,
        'connections': 100,
        'qps': 1000
    },
    target=SystemType.CONTAINER
)

# Generated Docker command:
# docker run --memory=3.0g --cpus=100
```

### 5. Application Configuration
Tunes thread pools, caches, and batch sizes:

```python
# Data processing application
config = advisor.analyze(
    workload_data={
        'working_set_gb': 50,
        'connections': 200,
        'batch_size': 10000
    },
    target=SystemType.APPLICATION
)

# Generated settings:
# thread_pool_size: 16         # Based on CPU cores
# connection_pool_size: 200    # Match concurrency
# cache_size: 229,739          # √n entries
# batch_size: 10,000           # Optimized for memory
```

## System Analysis

The advisor automatically profiles your system:

```python
from advisor import SystemAnalyzer

analyzer = SystemAnalyzer()
profile = analyzer.analyze_system()

print(f"CPU: {profile.cpu_count} cores ({profile.cpu_model})")
print(f"Memory: {profile.memory_gb:.1f}GB")
print(f"Storage: {profile.storage_type} ({profile.storage_iops} IOPS)")
print(f"L3 Cache: {profile.l3_cache_mb:.1f}MB")
```

## Workload Analysis

Characterize workloads from metrics or logs:

```python
from advisor import WorkloadAnalyzer

analyzer = WorkloadAnalyzer()

# From metrics
workload = analyzer.analyze_workload(metrics={
    'read_ratio': 0.8,
    'working_set_gb': 100,
    'qps': 10000,
    'connections': 500
})

# From logs
workload = analyzer.analyze_workload(logs=[
    "SELECT * FROM users WHERE id = 123",
    "UPDATE orders SET status = 'shipped'",
    # ... more log entries
])
```

## A/B Testing

Compare configurations scientifically:

```python
# Create two configurations
config_a = advisor.analyze(workload_a, target=SystemType.DATABASE)
config_b = advisor.analyze(workload_b, target=SystemType.DATABASE)

# Run A/B test
results = advisor.compare_configs(
    [config_a, config_b],
    test_duration=300  # 5 minutes
)

for result in results:
    print(f"{result.config_name}:")
    print(f"  Throughput: {result.metrics['throughput']} QPS")
    print(f"  Latency: {result.metrics['latency']} ms")
    print(f"  Winner: {'Yes' if result.winner else 'No'}")
```

## Export Configurations

Save configurations in appropriate formats:

```python
# PostgreSQL config file
advisor.export_config(db_config, "postgresql.conf")

# JVM startup script
advisor.export_config(jvm_config, "jvm_startup.sh")

# JSON for other systems
advisor.export_config(app_config, "app_config.json")
```

## √n Optimization Examples

The advisor applies Williams' space-time tradeoffs:

### Database Buffer Pool
For data larger than memory:
- Traditional: Try to cache everything (thrashing)
- √n approach: Cache √(data_size) for optimal performance
- Example: 1TB data → 32GB buffer pool (not 1TB!)

### JVM Young Generation
Balance GC frequency vs pause time:
- Traditional: Fixed percentage (25% of heap)
- √n approach: √(heap_size) for optimal GC
- Example: 64GB heap → 8GB young gen

### Application Cache
Limited memory for caching:
- Traditional: LRU with fixed size
- √n approach: √(total_items) cache entries
- Example: 1B items → 31,622 cache entries

## Real-World Impact

Organizations using these principles:
- **Google**: Bigtable uses √n buffer sizes
- **Facebook**: RocksDB applies similar concepts
- **PostgreSQL**: Shared buffers tuning
- **JVM**: G1GC uses √n heuristics
- **Linux**: Page cache management

## Advanced Usage

### Custom System Types

```python
class CustomConfigGenerator(ConfigurationGenerator):
    def generate_custom_config(self, system, workload):
        # Apply √n principles to your system
        buffer_size = self.sqrt_calc.calculate_optimal_buffer(
            workload.total_data_size_gb * 1024
        )
        return Configuration(...)
```

### Continuous Optimization

```python
# Monitor and adapt over time
while True:
    current_metrics = collect_metrics()
    
    if significant_change(current_metrics, last_metrics):
        new_config = advisor.analyze(
            workload_data=current_metrics,
            target=SystemType.DATABASE
        )
        apply_config(new_config)
    
    time.sleep(3600)  # Check hourly
```

## Examples

See [example_advisor.py](example_advisor.py) for comprehensive examples:
- PostgreSQL tuning for OLTP vs OLAP
- JVM configuration for latency vs throughput
- Container resource allocation
- Kernel tuning for different workloads
- A/B testing configurations
- Adaptive configuration over time

## Troubleshooting

### Memory Calculations
- Buffer sizes are capped at available memory
- √n sizing only applied when data > memory
- Consider OS overhead (typically 20% reserved)

### Performance Testing
- A/B tests simulate load (real tests needed)
- Confidence intervals require sufficient samples
- Network conditions affect distributed systems

## Future Enhancements

- Cloud provider specific configs (AWS, GCP, Azure)
- Kubernetes operator for automatic tuning
- Machine learning workload detection
- Integration with monitoring systems
- Automated rollback on regression

## See Also

- [SpaceTimeCore](../core/spacetime_core.py): √n calculations
- [Memory Profiler](../profiler/): Identify bottlenecks