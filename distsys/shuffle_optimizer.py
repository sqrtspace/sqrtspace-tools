#!/usr/bin/env python3
"""
Distributed Shuffle Optimizer: Optimize shuffle operations in distributed computing

Features:
- Buffer Sizing: Calculate optimal buffer sizes per node
- Spill Strategy: Decide when to spill based on memory pressure
- Aggregation Trees: Build √n-height aggregation trees
- Network Awareness: Consider network topology in optimization
- AI Explanations: Clear reasoning for optimization decisions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
import psutil
import socket
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import heapq
import zlib

# Import core components
from core.spacetime_core import (
    MemoryHierarchy,
    SqrtNCalculator,
    OptimizationStrategy,
    MemoryProfiler
)


class ShuffleStrategy(Enum):
    """Shuffle strategies for distributed systems"""
    ALL_TO_ALL = "all_to_all"              # Every node to every node
    TREE_AGGREGATE = "tree_aggregate"       # Hierarchical aggregation
    HASH_PARTITION = "hash_partition"       # Hash-based partitioning
    RANGE_PARTITION = "range_partition"     # Range-based partitioning
    COMBINER_BASED = "combiner_based"       # Local combining first


class CompressionType(Enum):
    """Compression algorithms for shuffle data"""
    NONE = "none"
    SNAPPY = "snappy"    # Fast, moderate compression
    ZLIB = "zlib"        # Slower, better compression
    LZ4 = "lz4"          # Very fast, light compression


@dataclass
class NodeInfo:
    """Information about a compute node"""
    node_id: str
    hostname: str
    cpu_cores: int
    memory_gb: float
    network_bandwidth_gbps: float
    storage_type: str  # 'ssd' or 'hdd'
    rack_id: Optional[str] = None


@dataclass
class ShuffleTask:
    """A shuffle task specification"""
    task_id: str
    input_partitions: int
    output_partitions: int
    data_size_gb: float
    key_distribution: str  # 'uniform', 'skewed', 'heavy_hitters'
    value_size_avg: int    # Average value size in bytes
    combiner_function: Optional[str] = None  # 'sum', 'max', 'collect', etc.


@dataclass
class ShufflePlan:
    """Optimized shuffle execution plan"""
    strategy: ShuffleStrategy
    buffer_sizes: Dict[str, int]  # node_id -> buffer_size
    spill_thresholds: Dict[str, float]  # node_id -> threshold
    aggregation_tree: Optional[Dict[str, List[str]]]  # parent -> children
    compression: CompressionType
    partition_assignment: Dict[int, str]  # partition -> node_id
    estimated_time: float
    estimated_network_usage: float
    memory_usage: Dict[str, float]
    explanation: str


@dataclass
class ShuffleMetrics:
    """Metrics from shuffle execution"""
    total_time: float
    network_bytes: int
    disk_spills: int
    memory_peak: int
    compression_ratio: float
    skew_factor: float  # Max/avg partition size


class NetworkTopology:
    """Model network topology for optimization"""
    
    def __init__(self, nodes: List[NodeInfo]):
        self.nodes = {n.node_id: n for n in nodes}
        self.racks = self._group_by_rack(nodes)
        self.bandwidth_matrix = self._build_bandwidth_matrix()
    
    def _group_by_rack(self, nodes: List[NodeInfo]) -> Dict[str, List[str]]:
        """Group nodes by rack"""
        racks = {}
        for node in nodes:
            rack = node.rack_id or 'default'
            if rack not in racks:
                racks[rack] = []
            racks[rack].append(node.node_id)
        return racks
    
    def _build_bandwidth_matrix(self) -> Dict[Tuple[str, str], float]:
        """Build bandwidth matrix between nodes"""
        matrix = {}
        for n1 in self.nodes:
            for n2 in self.nodes:
                if n1 == n2:
                    matrix[(n1, n2)] = float('inf')  # Local
                elif self._same_rack(n1, n2):
                    # Same rack: use min node bandwidth
                    matrix[(n1, n2)] = min(
                        self.nodes[n1].network_bandwidth_gbps,
                        self.nodes[n2].network_bandwidth_gbps
                    )
                else:
                    # Cross-rack: assume 50% of node bandwidth
                    matrix[(n1, n2)] = min(
                        self.nodes[n1].network_bandwidth_gbps,
                        self.nodes[n2].network_bandwidth_gbps
                    ) * 0.5
        return matrix
    
    def _same_rack(self, node1: str, node2: str) -> bool:
        """Check if two nodes are in the same rack"""
        r1 = self.nodes[node1].rack_id or 'default'
        r2 = self.nodes[node2].rack_id or 'default'
        return r1 == r2
    
    def get_bandwidth(self, src: str, dst: str) -> float:
        """Get bandwidth between two nodes in Gbps"""
        return self.bandwidth_matrix.get((src, dst), 1.0)


class CostModel:
    """Cost model for shuffle operations"""
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.hierarchy = MemoryHierarchy.detect_system()
    
    def estimate_shuffle_time(self, task: ShuffleTask, plan: ShufflePlan) -> float:
        """Estimate shuffle execution time"""
        # Network transfer time
        network_time = self._estimate_network_time(task, plan)
        
        # Disk I/O time (if spilling)
        io_time = self._estimate_io_time(task, plan)
        
        # CPU time (serialization, compression)
        cpu_time = self._estimate_cpu_time(task, plan)
        
        # Take max as they can overlap
        return max(network_time, io_time) + cpu_time * 0.1
    
    def _estimate_network_time(self, task: ShuffleTask, plan: ShufflePlan) -> float:
        """Estimate network transfer time"""
        bytes_per_partition = task.data_size_gb * 1e9 / task.input_partitions
        
        if plan.strategy == ShuffleStrategy.ALL_TO_ALL:
            # Every partition to every node
            total_bytes = task.data_size_gb * 1e9
            avg_bandwidth = np.mean(list(self.topology.bandwidth_matrix.values()))
            return total_bytes / (avg_bandwidth * 1e9)
        
        elif plan.strategy == ShuffleStrategy.TREE_AGGREGATE:
            # Log(n) levels in tree
            num_nodes = len(self.topology.nodes)
            tree_height = np.log2(num_nodes)
            bytes_per_level = task.data_size_gb * 1e9 / tree_height
            avg_bandwidth = np.mean(list(self.topology.bandwidth_matrix.values()))
            return tree_height * bytes_per_level / (avg_bandwidth * 1e9)
        
        else:
            # Hash/range partition: each partition to one node
            avg_bandwidth = np.mean(list(self.topology.bandwidth_matrix.values()))
            return bytes_per_partition * task.output_partitions / (avg_bandwidth * 1e9)
    
    def _estimate_io_time(self, task: ShuffleTask, plan: ShufflePlan) -> float:
        """Estimate disk I/O time if spilling"""
        total_spill = 0
        
        for node_id, threshold in plan.spill_thresholds.items():
            node = self.topology.nodes[node_id]
            buffer_size = plan.buffer_sizes[node_id]
            
            # Estimate spill amount
            node_data = task.data_size_gb * 1e9 / len(self.topology.nodes)
            if node_data > buffer_size:
                spill_amount = node_data - buffer_size
                total_spill += spill_amount
        
        if total_spill > 0:
            # Assume 200MB/s for HDD, 500MB/s for SSD
            io_speed = 500e6 if 'ssd' in str(plan).lower() else 200e6
            return total_spill / io_speed
        
        return 0.0
    
    def _estimate_cpu_time(self, task: ShuffleTask, plan: ShufflePlan) -> float:
        """Estimate CPU time for serialization and compression"""
        total_cores = sum(n.cpu_cores for n in self.topology.nodes.values())
        
        # Serialization cost
        serialize_rate = 1e9  # 1GB/s per core
        serialize_time = task.data_size_gb * 1e9 / (serialize_rate * total_cores)
        
        # Compression cost
        if plan.compression != CompressionType.NONE:
            if plan.compression == CompressionType.ZLIB:
                compress_rate = 100e6  # 100MB/s per core
            elif plan.compression == CompressionType.SNAPPY:
                compress_rate = 500e6  # 500MB/s per core
            else:  # LZ4
                compress_rate = 1e9    # 1GB/s per core
            
            compress_time = task.data_size_gb * 1e9 / (compress_rate * total_cores)
        else:
            compress_time = 0
        
        return serialize_time + compress_time


class ShuffleOptimizer:
    """Main distributed shuffle optimizer"""
    
    def __init__(self, nodes: List[NodeInfo], memory_limit_fraction: float = 0.5):
        self.topology = NetworkTopology(nodes)
        self.cost_model = CostModel(self.topology)
        self.memory_limit_fraction = memory_limit_fraction
        self.sqrt_calc = SqrtNCalculator()
    
    def optimize_shuffle(self, task: ShuffleTask) -> ShufflePlan:
        """Generate optimized shuffle plan"""
        # Choose strategy based on task characteristics
        strategy = self._choose_strategy(task)
        
        # Calculate buffer sizes using √n principle
        buffer_sizes = self._calculate_buffer_sizes(task)
        
        # Determine spill thresholds
        spill_thresholds = self._calculate_spill_thresholds(task, buffer_sizes)
        
        # Build aggregation tree if needed
        aggregation_tree = None
        if strategy == ShuffleStrategy.TREE_AGGREGATE:
            aggregation_tree = self._build_aggregation_tree()
        
        # Choose compression
        compression = self._choose_compression(task)
        
        # Assign partitions to nodes
        partition_assignment = self._assign_partitions(task, strategy)
        
        # Estimate performance
        plan = ShufflePlan(
            strategy=strategy,
            buffer_sizes=buffer_sizes,
            spill_thresholds=spill_thresholds,
            aggregation_tree=aggregation_tree,
            compression=compression,
            partition_assignment=partition_assignment,
            estimated_time=0.0,
            estimated_network_usage=0.0,
            memory_usage={},
            explanation=""
        )
        
        # Calculate estimates
        plan.estimated_time = self.cost_model.estimate_shuffle_time(task, plan)
        plan.estimated_network_usage = self._estimate_network_usage(task, plan)
        plan.memory_usage = self._estimate_memory_usage(task, plan)
        
        # Generate explanation
        plan.explanation = self._generate_explanation(task, plan)
        
        return plan
    
    def _choose_strategy(self, task: ShuffleTask) -> ShuffleStrategy:
        """Choose shuffle strategy based on task characteristics"""
        # Small data: all-to-all is fine
        if task.data_size_gb < 1:
            return ShuffleStrategy.ALL_TO_ALL
        
        # Has combiner: use combining strategy
        if task.combiner_function:
            return ShuffleStrategy.COMBINER_BASED
        
        # Many nodes: use tree aggregation
        if len(self.topology.nodes) > 10:
            return ShuffleStrategy.TREE_AGGREGATE
        
        # Skewed data: use range partitioning
        if task.key_distribution == 'skewed':
            return ShuffleStrategy.RANGE_PARTITION
        
        # Default: hash partitioning
        return ShuffleStrategy.HASH_PARTITION
    
    def _calculate_buffer_sizes(self, task: ShuffleTask) -> Dict[str, int]:
        """Calculate optimal buffer sizes using √n principle"""
        buffer_sizes = {}
        
        for node_id, node in self.topology.nodes.items():
            # Available memory for shuffle
            available_memory = node.memory_gb * 1e9 * self.memory_limit_fraction
            
            # Data size per node
            data_per_node = task.data_size_gb * 1e9 / len(self.topology.nodes)
            
            if data_per_node <= available_memory:
                # Can fit all data
                buffer_size = int(data_per_node)
            else:
                # Use √n buffer
                sqrt_buffer = self.sqrt_calc.calculate_interval(
                    int(data_per_node / task.value_size_avg)
                ) * task.value_size_avg
                buffer_size = min(int(sqrt_buffer), int(available_memory))
            
            buffer_sizes[node_id] = buffer_size
        
        return buffer_sizes
    
    def _calculate_spill_thresholds(self, task: ShuffleTask, 
                                  buffer_sizes: Dict[str, int]) -> Dict[str, float]:
        """Calculate memory thresholds for spilling"""
        thresholds = {}
        
        for node_id, buffer_size in buffer_sizes.items():
            # Spill at 80% of buffer to leave headroom
            thresholds[node_id] = buffer_size * 0.8
        
        return thresholds
    
    def _build_aggregation_tree(self) -> Dict[str, List[str]]:
        """Build √n-height aggregation tree"""
        nodes = list(self.topology.nodes.keys())
        n = len(nodes)
        
        # Calculate branching factor for √n height
        height = int(np.sqrt(n))
        branching_factor = int(np.ceil(n ** (1 / height)))
        
        tree = {}
        
        # Build tree level by level
        current_level = nodes[:]
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), branching_factor):
                # Group nodes
                group = current_level[i:i + branching_factor]
                if len(group) > 1:
                    parent = group[0]  # First node as parent
                    tree[parent] = group[1:]  # Rest as children
                    next_level.append(parent)
                elif group:
                    next_level.append(group[0])
            
            current_level = next_level
        
        return tree
    
    def _choose_compression(self, task: ShuffleTask) -> CompressionType:
        """Choose compression based on data characteristics and network"""
        # Average network bandwidth
        avg_bandwidth = np.mean([
            n.network_bandwidth_gbps for n in self.topology.nodes.values()
        ])
        
        # High bandwidth: no compression
        if avg_bandwidth > 10:  # 10+ Gbps
            return CompressionType.NONE
        
        # Large values: use better compression
        if task.value_size_avg > 1000:
            return CompressionType.ZLIB
        
        # Medium bandwidth: balanced compression
        if avg_bandwidth > 1:  # 1-10 Gbps
            return CompressionType.SNAPPY
        
        # Low bandwidth: fast compression
        return CompressionType.LZ4
    
    def _assign_partitions(self, task: ShuffleTask, 
                         strategy: ShuffleStrategy) -> Dict[int, str]:
        """Assign partitions to nodes"""
        nodes = list(self.topology.nodes.keys())
        assignment = {}
        
        if strategy == ShuffleStrategy.HASH_PARTITION:
            # Round-robin assignment
            for i in range(task.output_partitions):
                assignment[i] = nodes[i % len(nodes)]
        
        elif strategy == ShuffleStrategy.RANGE_PARTITION:
            # Assign ranges to nodes
            partitions_per_node = task.output_partitions // len(nodes)
            for i, node in enumerate(nodes):
                start = i * partitions_per_node
                end = start + partitions_per_node
                if i == len(nodes) - 1:
                    end = task.output_partitions
                for p in range(start, end):
                    assignment[p] = node
        
        else:
            # Default: even distribution
            for i in range(task.output_partitions):
                assignment[i] = nodes[i % len(nodes)]
        
        return assignment
    
    def _estimate_network_usage(self, task: ShuffleTask, plan: ShufflePlan) -> float:
        """Estimate total network bytes"""
        base_bytes = task.data_size_gb * 1e9
        
        # Apply compression ratio
        if plan.compression == CompressionType.ZLIB:
            base_bytes *= 0.3  # ~70% compression
        elif plan.compression == CompressionType.SNAPPY:
            base_bytes *= 0.5  # ~50% compression
        elif plan.compression == CompressionType.LZ4:
            base_bytes *= 0.7  # ~30% compression
        
        # Apply strategy multiplier
        if plan.strategy == ShuffleStrategy.ALL_TO_ALL:
            n = len(self.topology.nodes)
            base_bytes *= (n - 1) / n  # Each node sends to n-1 others
        elif plan.strategy == ShuffleStrategy.TREE_AGGREGATE:
            # Log(n) levels
            base_bytes *= np.log2(len(self.topology.nodes))
        
        return base_bytes
    
    def _estimate_memory_usage(self, task: ShuffleTask, plan: ShufflePlan) -> Dict[str, float]:
        """Estimate memory usage per node"""
        memory_usage = {}
        
        for node_id in self.topology.nodes:
            # Buffer memory
            buffer_mem = plan.buffer_sizes[node_id]
            
            # Overhead (metadata, indices)
            overhead = buffer_mem * 0.1
            
            # Compression buffers if used
            compress_mem = 0
            if plan.compression != CompressionType.NONE:
                compress_mem = min(buffer_mem * 0.1, 100 * 1024 * 1024)  # Max 100MB
            
            memory_usage[node_id] = buffer_mem + overhead + compress_mem
        
        return memory_usage
    
    def _generate_explanation(self, task: ShuffleTask, plan: ShufflePlan) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Strategy explanation
        strategy_reasons = {
            ShuffleStrategy.ALL_TO_ALL: "small data size allows full exchange",
            ShuffleStrategy.TREE_AGGREGATE: f"√n-height tree reduces network hops to {int(np.sqrt(len(self.topology.nodes)))}",
            ShuffleStrategy.HASH_PARTITION: "uniform data distribution suits hash partitioning",
            ShuffleStrategy.RANGE_PARTITION: "skewed data benefits from range partitioning",
            ShuffleStrategy.COMBINER_BASED: "combiner function enables local aggregation"
        }
        
        explanations.append(
            f"Using {plan.strategy.value} strategy because {strategy_reasons[plan.strategy]}."
        )
        
        # Buffer sizing
        avg_buffer_mb = np.mean(list(plan.buffer_sizes.values())) / 1e6
        explanations.append(
            f"Allocated {avg_buffer_mb:.0f}MB buffers per node using √n principle "
            f"to balance memory usage and I/O."
        )
        
        # Compression
        if plan.compression != CompressionType.NONE:
            explanations.append(
                f"Applied {plan.compression.value} compression to reduce network "
                f"traffic by ~{(1 - plan.estimated_network_usage / (task.data_size_gb * 1e9)) * 100:.0f}%."
            )
        
        # Performance estimate
        explanations.append(
            f"Estimated completion time: {plan.estimated_time:.1f}s with "
            f"{plan.estimated_network_usage / 1e9:.1f}GB network transfer."
        )
        
        return " ".join(explanations)
    
    def execute_shuffle(self, task: ShuffleTask, plan: ShufflePlan) -> ShuffleMetrics:
        """Simulate shuffle execution (for testing)"""
        start_time = time.time()
        
        # Simulate execution
        time.sleep(0.1)  # Simulate some work
        
        # Calculate metrics
        metrics = ShuffleMetrics(
            total_time=time.time() - start_time,
            network_bytes=int(plan.estimated_network_usage),
            disk_spills=sum(1 for b in plan.buffer_sizes.values() 
                          if b < task.data_size_gb * 1e9 / len(self.topology.nodes)),
            memory_peak=max(plan.memory_usage.values()),
            compression_ratio=1.0,
            skew_factor=1.0
        )
        
        if plan.compression == CompressionType.ZLIB:
            metrics.compression_ratio = 3.3
        elif plan.compression == CompressionType.SNAPPY:
            metrics.compression_ratio = 2.0
        elif plan.compression == CompressionType.LZ4:
            metrics.compression_ratio = 1.4
        
        return metrics


def create_test_cluster(num_nodes: int = 4) -> List[NodeInfo]:
    """Create a test cluster configuration"""
    nodes = []
    
    for i in range(num_nodes):
        node = NodeInfo(
            node_id=f"node{i}",
            hostname=f"worker{i}.cluster.local",
            cpu_cores=16,
            memory_gb=64,
            network_bandwidth_gbps=10.0,
            storage_type='ssd',
            rack_id=f"rack{i // 2}"  # 2 nodes per rack
        )
        nodes.append(node)
    
    return nodes


# Example usage
if __name__ == "__main__":
    print("Distributed Shuffle Optimizer Example")
    print("="*60)
    
    # Create test cluster
    nodes = create_test_cluster(4)
    optimizer = ShuffleOptimizer(nodes)
    
    # Example 1: Small uniform shuffle
    print("\nExample 1: Small uniform shuffle")
    task1 = ShuffleTask(
        task_id="shuffle_1",
        input_partitions=100,
        output_partitions=100,
        data_size_gb=0.5,
        key_distribution='uniform',
        value_size_avg=100
    )
    
    plan1 = optimizer.optimize_shuffle(task1)
    print(f"Strategy: {plan1.strategy.value}")
    print(f"Compression: {plan1.compression.value}")
    print(f"Estimated time: {plan1.estimated_time:.2f}s")
    print(f"Explanation: {plan1.explanation}")
    
    # Example 2: Large skewed shuffle
    print("\n\nExample 2: Large skewed shuffle")
    task2 = ShuffleTask(
        task_id="shuffle_2",
        input_partitions=1000,
        output_partitions=500,
        data_size_gb=100,
        key_distribution='skewed',
        value_size_avg=1000,
        combiner_function='sum'
    )
    
    plan2 = optimizer.optimize_shuffle(task2)
    print(f"Strategy: {plan2.strategy.value}")
    print(f"Buffer sizes: {list(plan2.buffer_sizes.values())[0] / 1e9:.1f}GB per node")
    print(f"Network usage: {plan2.estimated_network_usage / 1e9:.1f}GB")
    print(f"Explanation: {plan2.explanation}")
    
    # Example 3: Many nodes with aggregation
    print("\n\nExample 3: Many nodes with tree aggregation")
    large_cluster = create_test_cluster(16)
    large_optimizer = ShuffleOptimizer(large_cluster)
    
    task3 = ShuffleTask(
        task_id="shuffle_3",
        input_partitions=10000,
        output_partitions=16,
        data_size_gb=50,
        key_distribution='uniform',
        value_size_avg=200,
        combiner_function='collect'
    )
    
    plan3 = large_optimizer.optimize_shuffle(task3)
    print(f"Strategy: {plan3.strategy.value}")
    if plan3.aggregation_tree:
        print(f"Tree height: {int(np.sqrt(len(large_cluster)))}")
        print(f"Tree structure sample: {list(plan3.aggregation_tree.items())[:3]}")
    print(f"Explanation: {plan3.explanation}")
    
    # Simulate execution
    print("\n\nSimulating shuffle execution...")
    metrics = optimizer.execute_shuffle(task1, plan1)
    print(f"Execution time: {metrics.total_time:.3f}s")
    print(f"Network bytes: {metrics.network_bytes / 1e6:.1f}MB")
    print(f"Compression ratio: {metrics.compression_ratio:.1f}x")
