#!/usr/bin/env python3
"""
Example demonstrating Cache-Aware Data Structures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_aware_structures import (
    AdaptiveMap,
    CompressedTrie,
    create_optimized_structure,
    MemoryHierarchy
)
import time
import random
import string


def demonstrate_adaptive_behavior():
    """Show how AdaptiveMap adapts to different sizes"""
    print("="*60)
    print("Adaptive Map Behavior")
    print("="*60)
    
    # Create adaptive map
    amap = AdaptiveMap[int, str]()
    
    # Track adaptations
    print("\nInserting data and watching adaptations:")
    print("-" * 50)
    
    sizes = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
    
    for target_size in sizes:
        # Insert to reach target size
        current = amap.size()
        for i in range(current, target_size):
            amap.put(i, f"value_{i}")
        
        stats = amap.get_stats()
        if stats['size'] in sizes:  # Only print at milestones
            print(f"Size: {stats['size']:>6} | "
                  f"Implementation: {stats['implementation']:>10} | "
                  f"Memory: {stats['memory_level']:>5}")
    
    # Test different access patterns
    print("\n\nTesting access patterns:")
    print("-" * 50)
    
    # Sequential access
    print("Sequential access pattern...")
    for i in range(100):
        amap.get(i)
    
    stats = amap.get_stats()
    print(f"  Sequential ratio: {stats['access_pattern']['sequential_ratio']:.2f}")
    
    # Random access
    print("\nRandom access pattern...")
    for _ in range(100):
        amap.get(random.randint(0, 999))
    
    stats = amap.get_stats()
    print(f"  Sequential ratio: {stats['access_pattern']['sequential_ratio']:.2f}")


def benchmark_structures():
    """Compare performance of different structures"""
    print("\n\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    sizes = [100, 1000, 10000, 100000]
    
    print(f"\n{'Size':>8} | {'Dict':>8} | {'Adaptive':>8} | {'Speedup':>8}")
    print("-" * 40)
    
    for n in sizes:
        # Generate test data
        keys = [f"key_{i:06d}" for i in range(n)]
        values = [f"value_{i}" for i in range(n)]
        
        # Benchmark standard dict
        start = time.time()
        std_dict = {}
        for k, v in zip(keys, values):
            std_dict[k] = v
        for k in keys[:1000]:  # Sample lookups
            _ = std_dict.get(k)
        dict_time = time.time() - start
        
        # Benchmark adaptive map
        start = time.time()
        adaptive = AdaptiveMap[str, str]()
        for k, v in zip(keys, values):
            adaptive.put(k, v)
        for k in keys[:1000]:  # Sample lookups
            _ = adaptive.get(k)
        adaptive_time = time.time() - start
        
        speedup = dict_time / adaptive_time
        print(f"{n:>8} | {dict_time:>8.3f} | {adaptive_time:>8.3f} | {speedup:>8.2f}x")


def demonstrate_cache_optimization():
    """Show cache line optimization benefits"""
    print("\n\n" + "="*60)
    print("Cache Line Optimization")
    print("="*60)
    
    hierarchy = MemoryHierarchy.detect_system()
    cache_line_size = 64
    
    print(f"\nSystem Information:")
    print(f"  Cache line size: {cache_line_size} bytes")
    print(f"  L1 cache: {hierarchy.l1_size / 1024:.0f}KB")
    print(f"  L2 cache: {hierarchy.l2_size / 1024:.0f}KB")
    print(f"  L3 cache: {hierarchy.l3_size / 1024 / 1024:.1f}MB")
    
    # Calculate optimal parameters
    print(f"\nOptimal Structure Parameters:")
    
    # For different key/value sizes
    configs = [
        ("Small (4B key, 4B value)", 4, 4),
        ("Medium (8B key, 8B value)", 8, 8),
        ("Large (16B key, 32B value)", 16, 32),
    ]
    
    for name, key_size, value_size in configs:
        entry_size = key_size + value_size
        entries_per_line = cache_line_size // entry_size
        
        # B-tree node size
        btree_keys = entries_per_line - 1  # Leave room for child pointers
        
        # Hash table bucket
        hash_entries = cache_line_size // entry_size
        
        print(f"\n{name}:")
        print(f"  Entries per cache line: {entries_per_line}")
        print(f"  B-tree keys per node: {btree_keys}")
        print(f"  Hash bucket capacity: {hash_entries}")
        
        # Calculate memory efficiency
        utilization = (entries_per_line * entry_size) / cache_line_size * 100
        print(f"  Cache utilization: {utilization:.1f}%")


def demonstrate_compressed_trie():
    """Show compressed trie benefits for strings"""
    print("\n\n" + "="*60)
    print("Compressed Trie for String Data")
    print("="*60)
    
    # Create trie
    trie = CompressedTrie()
    
    # Common prefixes scenario (URLs, file paths, etc.)
    test_data = [
        # API endpoints
        ("/api/v1/users/list", "list_users"),
        ("/api/v1/users/get", "get_user"),
        ("/api/v1/users/create", "create_user"),
        ("/api/v1/users/update", "update_user"),
        ("/api/v1/users/delete", "delete_user"),
        ("/api/v1/products/list", "list_products"),
        ("/api/v1/products/get", "get_product"),
        ("/api/v2/users/list", "list_users_v2"),
        ("/api/v2/analytics/events", "analytics_events"),
        ("/api/v2/analytics/metrics", "analytics_metrics"),
    ]
    
    print("\nInserting API endpoints:")
    for path, handler in test_data:
        trie.insert(path, handler)
        print(f"  {path} -> {handler}")
    
    # Memory comparison
    print("\n\nMemory Comparison:")
    
    # Trie size estimation (simplified)
    trie_nodes = 50  # Approximate with compression
    trie_memory = trie_nodes * 64  # 64 bytes per node
    
    # Dict size
    dict_memory = len(test_data) * (50 + 20) * 2  # key + value + overhead
    
    print(f"  Standard dict: ~{dict_memory} bytes")
    print(f"  Compressed trie: ~{trie_memory} bytes")
    print(f"  Compression ratio: {dict_memory / trie_memory:.1f}x")
    
    # Search demonstration
    print("\n\nSearching:")
    search_keys = [
        "/api/v1/users/list",
        "/api/v2/analytics/events",
        "/api/v3/users/list",  # Not found
    ]
    
    for key in search_keys:
        result = trie.search(key)
        status = "Found" if result else "Not found"
        print(f"  {key}: {status} {f'-> {result}' if result else ''}")


def demonstrate_external_memory():
    """Show external memory map with √n buffers"""
    print("\n\n" + "="*60)
    print("External Memory Map (Disk-backed)")
    print("="*60)
    
    # Create external map with explicit hint
    emap = create_optimized_structure(
        hint_type='external',
        hint_memory_limit=1024*1024  # 1MB buffer limit
    )
    
    print("\nSimulating large dataset that doesn't fit in memory:")
    
    # Insert large dataset
    n = 1000000  # 1M entries
    print(f"  Dataset size: {n:,} entries")
    print(f"  Estimated size: {n * 20 / 1e6:.1f}MB")
    
    # Buffer size calculation
    sqrt_n = int(n ** 0.5)
    buffer_entries = sqrt_n
    buffer_memory = buffer_entries * 20  # 20 bytes per entry
    
    print(f"\n√n Buffer Configuration:")
    print(f"  Buffer entries: {buffer_entries:,} (√{n:,})")
    print(f"  Buffer memory: {buffer_memory / 1024:.1f}KB")
    print(f"  Memory reduction: {(1 - sqrt_n/n) * 100:.1f}%")
    
    # Simulate access patterns
    print(f"\n\nAccess Pattern Analysis:")
    
    # Sequential scan
    sequential_hits = 0
    for i in range(1000):
        # Simulate buffer hit/miss
        if i % sqrt_n < 100:  # In buffer
            sequential_hits += 1
    
    print(f"  Sequential scan: {sequential_hits/10:.1f}% buffer hit rate")
    
    # Random access
    random_hits = 0
    for _ in range(1000):
        i = random.randint(0, n-1)
        if random.random() < sqrt_n/n:  # Probability in buffer
            random_hits += 1
    
    print(f"  Random access: {random_hits/10:.1f}% buffer hit rate")
    
    # Recommendations
    print(f"\n\nRecommendations:")
    print(f"  - Use sequential access when possible (better cache hits)")
    print(f"  - Group related keys together (spatial locality)")
    print(f"  - Consider compression for values (reduce I/O)")


def main():
    """Run all demonstrations"""
    demonstrate_adaptive_behavior()
    benchmark_structures()
    demonstrate_cache_optimization()
    demonstrate_compressed_trie()
    demonstrate_external_memory()
    
    print("\n\n" + "="*60)
    print("Cache-Aware Data Structures Complete!")
    print("="*60)
    print("\nKey Takeaways:")
    print("- Structures adapt to data size automatically")
    print("- Cache line alignment improves performance")
    print("- √n buffers enable huge datasets with limited memory")
    print("- Compression trades CPU for memory")
    print("="*60)


if __name__ == "__main__":
    main()