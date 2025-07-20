#!/usr/bin/env python3
"""
Example demonstrating Distributed Shuffle Optimizer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shuffle_optimizer import (
    ShuffleOptimizer,
    ShuffleTask,
    NodeInfo,
    create_test_cluster
)
import numpy as np


def demonstrate_basic_shuffle():
    """Basic shuffle optimization demonstration"""
    print("="*60)
    print("Basic Shuffle Optimization")
    print("="*60)
    
    # Create a 4-node cluster
    nodes = create_test_cluster(4)
    optimizer = ShuffleOptimizer(nodes)
    
    print("\nCluster configuration:")
    for node in nodes:
        print(f"  {node.node_id}: {node.cpu_cores} cores, "
              f"{node.memory_gb}GB RAM, {node.network_bandwidth_gbps}Gbps")
    
    # Simple shuffle task
    task = ShuffleTask(
        task_id="wordcount_shuffle",
        input_partitions=100,
        output_partitions=50,
        data_size_gb=10,
        key_distribution='uniform',
        value_size_avg=50,  # Small values (word counts)
        combiner_function='sum'
    )
    
    print(f"\nShuffle task:")
    print(f"  Input: {task.input_partitions} partitions, {task.data_size_gb}GB")
    print(f"  Output: {task.output_partitions} partitions")
    print(f"  Distribution: {task.key_distribution}")
    
    # Optimize
    plan = optimizer.optimize_shuffle(task)
    
    print(f"\nOptimization results:")
    print(f"  Strategy: {plan.strategy.value}")
    print(f"  Compression: {plan.compression.value}")
    print(f"  Buffer size: {list(plan.buffer_sizes.values())[0] / 1e6:.0f}MB per node")
    print(f"  Estimated time: {plan.estimated_time:.1f}s")
    print(f"  Network transfer: {plan.estimated_network_usage / 1e9:.1f}GB")
    print(f"\nExplanation: {plan.explanation}")


def demonstrate_large_scale_shuffle():
    """Large-scale shuffle with many nodes"""
    print("\n\n" + "="*60)
    print("Large-Scale Shuffle (32 nodes)")
    print("="*60)
    
    # Create larger cluster
    nodes = []
    for i in range(32):
        node = NodeInfo(
            node_id=f"node{i:02d}",
            hostname=f"worker{i}.bigcluster.local",
            cpu_cores=32,
            memory_gb=128,
            network_bandwidth_gbps=25.0,  # High-speed network
            storage_type='ssd',
            rack_id=f"rack{i // 8}"  # 8 nodes per rack
        )
        nodes.append(node)
    
    optimizer = ShuffleOptimizer(nodes, memory_limit_fraction=0.4)
    
    print(f"\nCluster: 32 nodes across {len(set(n.rack_id for n in nodes))} racks")
    print(f"Total resources: {sum(n.cpu_cores for n in nodes)} cores, "
          f"{sum(n.memory_gb for n in nodes)}GB RAM")
    
    # Large shuffle task (e.g., distributed sort)
    task = ShuffleTask(
        task_id="terasort_shuffle",
        input_partitions=10000,
        output_partitions=10000,
        data_size_gb=1000,  # 1TB shuffle
        key_distribution='uniform',
        value_size_avg=100
    )
    
    print(f"\nShuffle task: 1TB distributed sort")
    print(f"  {task.input_partitions} → {task.output_partitions} partitions")
    
    # Optimize
    plan = optimizer.optimize_shuffle(task)
    
    print(f"\nOptimization results:")
    print(f"  Strategy: {plan.strategy.value}")
    print(f"  Compression: {plan.compression.value}")
    
    # Show buffer calculation
    data_per_node = task.data_size_gb / len(nodes)
    buffer_per_node = list(plan.buffer_sizes.values())[0] / 1e9
    
    print(f"\nMemory management:")
    print(f"  Data per node: {data_per_node:.1f}GB")
    print(f"  Buffer per node: {buffer_per_node:.1f}GB")
    print(f"  Buffer ratio: {buffer_per_node / data_per_node:.2f}")
    
    # Check if using √n optimization
    if buffer_per_node < data_per_node * 0.5:
        print(f"  ✓ Using √n buffers to save memory")
    
    print(f"\nPerformance estimates:")
    print(f"  Time: {plan.estimated_time:.0f}s ({plan.estimated_time/60:.1f} minutes)")
    print(f"  Network: {plan.estimated_network_usage / 1e12:.2f}TB")
    
    # Show aggregation tree structure
    if plan.aggregation_tree:
        print(f"\nAggregation tree:")
        print(f"  Height: {int(np.sqrt(len(nodes)))} levels")
        print(f"  Fanout: ~{len(nodes) ** (1/int(np.sqrt(len(nodes)))):.0f} nodes per level")


def demonstrate_skewed_data():
    """Handling skewed data distribution"""
    print("\n\n" + "="*60)
    print("Skewed Data Optimization")
    print("="*60)
    
    nodes = create_test_cluster(8)
    optimizer = ShuffleOptimizer(nodes)
    
    # Skewed shuffle (e.g., popular keys in recommendation system)
    task = ShuffleTask(
        task_id="recommendation_shuffle",
        input_partitions=1000,
        output_partitions=100,
        data_size_gb=50,
        key_distribution='skewed',  # Some keys much more frequent
        value_size_avg=500,  # User profiles
        combiner_function='collect'
    )
    
    print(f"\nSkewed shuffle scenario:")
    print(f"  Use case: User recommendation aggregation")
    print(f"  Problem: Some users have many more interactions")
    print(f"  Data: {task.data_size_gb}GB with skewed distribution")
    
    # Optimize
    plan = optimizer.optimize_shuffle(task)
    
    print(f"\nOptimization for skewed data:")
    print(f"  Strategy: {plan.strategy.value}")
    print(f"  Reason: Handles data skew better than hash partitioning")
    
    # Show partition assignment
    print(f"\nPartition distribution:")
    nodes_with_partitions = {}
    for partition, node in plan.partition_assignment.items():
        if node not in nodes_with_partitions:
            nodes_with_partitions[node] = 0
        nodes_with_partitions[node] += 1
    
    for node, count in sorted(nodes_with_partitions.items())[:4]:
        print(f"  {node}: {count} partitions")
    
    print(f"\n{plan.explanation}")


def demonstrate_memory_pressure():
    """Optimization under memory pressure"""
    print("\n\n" + "="*60)
    print("Memory-Constrained Shuffle")
    print("="*60)
    
    # Create memory-constrained cluster
    nodes = []
    for i in range(4):
        node = NodeInfo(
            node_id=f"small_node{i}",
            hostname=f"micro{i}.local",
            cpu_cores=4,
            memory_gb=8,  # Only 8GB RAM
            network_bandwidth_gbps=1.0,  # Slow network
            storage_type='hdd'  # Slower storage
        )
        nodes.append(node)
    
    # Use only 30% of memory for shuffle
    optimizer = ShuffleOptimizer(nodes, memory_limit_fraction=0.3)
    
    print(f"\nResource-constrained cluster:")
    print(f"  4 nodes with 8GB RAM each")
    print(f"  Only 30% memory available for shuffle")
    print(f"  Slow network (1Gbps) and HDD storage")
    
    # Large shuffle relative to resources
    task = ShuffleTask(
        task_id="constrained_shuffle",
        input_partitions=1000,
        output_partitions=1000,
        data_size_gb=100,  # 100GB with only 32GB total RAM
        key_distribution='uniform',
        value_size_avg=1000
    )
    
    print(f"\nChallenge: Shuffle {task.data_size_gb}GB with {sum(n.memory_gb for n in nodes)}GB total RAM")
    
    # Optimize
    plan = optimizer.optimize_shuffle(task)
    
    print(f"\nMemory optimization:")
    buffer_mb = list(plan.buffer_sizes.values())[0] / 1e6
    spill_threshold_mb = list(plan.spill_thresholds.values())[0] / 1e6
    
    print(f"  Buffer size: {buffer_mb:.0f}MB per node")
    print(f"  Spill threshold: {spill_threshold_mb:.0f}MB")
    print(f"  Compression: {plan.compression.value} (reduces memory pressure)")
    
    # Calculate spill statistics
    data_per_node = task.data_size_gb * 1e9 / len(nodes)
    buffer_size = list(plan.buffer_sizes.values())[0]
    spill_ratio = max(0, (data_per_node - buffer_size) / data_per_node)
    
    print(f"\nSpill analysis:")
    print(f"  Data per node: {data_per_node / 1e9:.1f}GB")
    print(f"  Must spill: {spill_ratio * 100:.0f}% to disk")
    print(f"  I/O overhead: ~{spill_ratio * plan.estimated_time:.0f}s")
    
    print(f"\n{plan.explanation}")


def demonstrate_adaptive_optimization():
    """Show how optimization adapts to different scenarios"""
    print("\n\n" + "="*60)
    print("Adaptive Optimization Comparison")
    print("="*60)
    
    nodes = create_test_cluster(8)
    optimizer = ShuffleOptimizer(nodes)
    
    scenarios = [
        ("Small data", ShuffleTask("s1", 10, 10, 0.1, 'uniform', 100)),
        ("Large uniform", ShuffleTask("s2", 1000, 1000, 100, 'uniform', 100)),
        ("Skewed with combiner", ShuffleTask("s3", 1000, 100, 50, 'skewed', 200, 'sum')),
        ("Wide shuffle", ShuffleTask("s4", 100, 1000, 10, 'uniform', 50)),
    ]
    
    print(f"\nComparing optimization strategies:")
    print(f"{'Scenario':<20} {'Data':>8} {'Strategy':<20} {'Compression':<12} {'Time':>8}")
    print("-" * 80)
    
    for name, task in scenarios:
        plan = optimizer.optimize_shuffle(task)
        print(f"{name:<20} {task.data_size_gb:>6.1f}GB "
              f"{plan.strategy.value:<20} {plan.compression.value:<12} "
              f"{plan.estimated_time:>6.1f}s")
    
    print("\nKey insights:")
    print("- Small data uses all-to-all (simple and fast)")
    print("- Large uniform data uses hash partitioning")
    print("- Skewed data with combiner uses combining strategy")
    print("- Compression chosen based on network bandwidth")


def main():
    """Run all demonstrations"""
    demonstrate_basic_shuffle()
    demonstrate_large_scale_shuffle()
    demonstrate_skewed_data()
    demonstrate_memory_pressure()
    demonstrate_adaptive_optimization()
    
    print("\n" + "="*60)
    print("Distributed Shuffle Optimization Complete!")
    print("="*60)


if __name__ == "__main__":
    main()