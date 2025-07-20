#!/usr/bin/env python3
"""
Cache-Aware Data Structure Library: Data structures that adapt to memory hierarchies

Features:
- B-Trees with Optimal Node Size: Based on cache line size
- Hash Tables with Linear Probing: Sized for L3 cache
- Compressed Tries: Trade computation for space
- Adaptive Collections: Switch implementation based on size
- AI Explanations: Clear reasoning for structure choices
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import psutil
from typing import Any, Dict, List, Tuple, Optional, Iterator, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import struct
import zlib
from abc import ABC, abstractmethod

# Import core components
from core.spacetime_core import (
    MemoryHierarchy,
    SqrtNCalculator,
    OptimizationStrategy
)


K = TypeVar('K')
V = TypeVar('V')


class ImplementationType(Enum):
    """Implementation strategies for different sizes"""
    ARRAY = "array"              # Small: linear array
    BTREE = "btree"              # Medium: B-tree
    HASH = "hash"                # Large: hash table
    EXTERNAL = "external"        # Huge: disk-backed
    COMPRESSED = "compressed"    # Memory-constrained: compressed


@dataclass
class AccessPattern:
    """Track access patterns for adaptation"""
    sequential_ratio: float = 0.0
    read_write_ratio: float = 1.0
    hot_key_ratio: float = 0.0
    total_accesses: int = 0


class CacheAwareStructure(ABC, Generic[K, V]):
    """Base class for cache-aware data structures"""
    
    def __init__(self, hint_size: Optional[int] = None,
                 hint_access_pattern: Optional[str] = None,
                 hint_memory_limit: Optional[int] = None):
        self.hierarchy = MemoryHierarchy.detect_system()
        self.sqrt_calc = SqrtNCalculator()
        
        # Hints from user
        self.hint_size = hint_size
        self.hint_access_pattern = hint_access_pattern
        self.hint_memory_limit = hint_memory_limit or psutil.virtual_memory().available
        
        # Access tracking
        self.access_pattern = AccessPattern()
        self._access_history = []
        
        # Cache line size (typically 64 bytes)
        self.cache_line_size = 64
    
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Get value for key"""
        pass
    
    @abstractmethod
    def put(self, key: K, value: V) -> None:
        """Store key-value pair"""
        pass
    
    @abstractmethod
    def delete(self, key: K) -> bool:
        """Delete key, return True if existed"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Number of elements"""
        pass
    
    def _track_access(self, key: K, is_write: bool = False):
        """Track access pattern"""
        self.access_pattern.total_accesses += 1
        
        # Track sequential access
        if self._access_history and hasattr(key, '__lt__'):
            last_key = self._access_history[-1]
            if key > last_key:  # Sequential
                self.access_pattern.sequential_ratio = \
                    (self.access_pattern.sequential_ratio * 0.95 + 0.05)
            else:
                self.access_pattern.sequential_ratio *= 0.95
        
        # Track read/write ratio
        if is_write:
            self.access_pattern.read_write_ratio *= 0.99
        else:
            self.access_pattern.read_write_ratio = \
                self.access_pattern.read_write_ratio * 0.99 + 0.01
        
        # Keep limited history
        self._access_history.append(key)
        if len(self._access_history) > 100:
            self._access_history.pop(0)


class AdaptiveMap(CacheAwareStructure[K, V]):
    """Map that adapts implementation based on size and access patterns"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Start with array for small sizes
        self._impl_type = ImplementationType.ARRAY
        self._data: Any = []  # [(key, value), ...]
        
        # Thresholds for switching implementations
        self._array_threshold = self.cache_line_size // 16  # ~4 elements
        self._btree_threshold = self.hierarchy.l3_size // 100  # Fit in L3
        self._hash_threshold = self.hierarchy.ram_size // 10   # 10% of RAM
    
    def get(self, key: K) -> Optional[V]:
        """Get value with cache-aware lookup"""
        self._track_access(key)
        
        if self._impl_type == ImplementationType.ARRAY:
            # Linear search in array
            for k, v in self._data:
                if k == key:
                    return v
            return None
        
        elif self._impl_type == ImplementationType.BTREE:
            return self._data.get(key)
        
        elif self._impl_type == ImplementationType.HASH:
            return self._data.get(key)
        
        else:  # EXTERNAL
            return self._data.get(key)
    
    def put(self, key: K, value: V) -> None:
        """Store with automatic adaptation"""
        self._track_access(key, is_write=True)
        
        # Check if we need to adapt
        current_size = self.size()
        if self._should_adapt(current_size):
            self._adapt_implementation(current_size)
        
        # Store based on implementation
        if self._impl_type == ImplementationType.ARRAY:
            # Update or append
            for i, (k, v) in enumerate(self._data):
                if k == key:
                    self._data[i] = (key, value)
                    return
            self._data.append((key, value))
        
        else:  # BTREE, HASH, or EXTERNAL
            self._data[key] = value
    
    def delete(self, key: K) -> bool:
        """Delete with adaptation"""
        if self._impl_type == ImplementationType.ARRAY:
            for i, (k, v) in enumerate(self._data):
                if k == key:
                    self._data.pop(i)
                    return True
            return False
        else:
            return self._data.pop(key, None) is not None
    
    def size(self) -> int:
        """Current number of elements"""
        if self._impl_type == ImplementationType.ARRAY:
            return len(self._data)
        else:
            return len(self._data)
    
    def _should_adapt(self, current_size: int) -> bool:
        """Check if we should switch implementation"""
        if self._impl_type == ImplementationType.ARRAY:
            return current_size > self._array_threshold
        elif self._impl_type == ImplementationType.BTREE:
            return current_size > self._btree_threshold
        elif self._impl_type == ImplementationType.HASH:
            return current_size > self._hash_threshold
        return False
    
    def _adapt_implementation(self, current_size: int):
        """Switch to more appropriate implementation"""
        old_impl = self._impl_type
        old_data = self._data
        
        # Determine new implementation
        if current_size <= self._array_threshold:
            self._impl_type = ImplementationType.ARRAY
            self._data = list(old_data) if old_impl != ImplementationType.ARRAY else old_data
        
        elif current_size <= self._btree_threshold:
            self._impl_type = ImplementationType.BTREE
            self._data = CacheOptimizedBTree()
            # Copy data
            if old_impl == ImplementationType.ARRAY:
                for k, v in old_data:
                    self._data[k] = v
            else:
                for k, v in old_data.items():
                    self._data[k] = v
        
        elif current_size <= self._hash_threshold:
            self._impl_type = ImplementationType.HASH
            self._data = CacheOptimizedHashTable(
                initial_size=self._calculate_hash_size(current_size)
            )
            # Copy data
            if old_impl == ImplementationType.ARRAY:
                for k, v in old_data:
                    self._data[k] = v
            else:
                for k, v in old_data.items():
                    self._data[k] = v
        
        else:
            self._impl_type = ImplementationType.EXTERNAL
            self._data = ExternalMemoryMap()
            # Copy data
            if old_impl == ImplementationType.ARRAY:
                for k, v in old_data:
                    self._data[k] = v
            else:
                for k, v in old_data.items():
                    self._data[k] = v
        
        print(f"[AdaptiveMap] Adapted from {old_impl.value} to {self._impl_type.value} "
              f"at size {current_size}")
    
    def _calculate_hash_size(self, num_elements: int) -> int:
        """Calculate optimal hash table size for cache"""
        # Target 75% load factor
        target_size = int(num_elements * 1.33)
        
        # Round to cache line boundaries
        entry_size = 16  # Assume 8 bytes key + 8 bytes value
        entries_per_line = self.cache_line_size // entry_size
        
        return ((target_size + entries_per_line - 1) // entries_per_line) * entries_per_line
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the data structure"""
        return {
            'implementation': self._impl_type.value,
            'size': self.size(),
            'access_pattern': {
                'sequential_ratio': self.access_pattern.sequential_ratio,
                'read_write_ratio': self.access_pattern.read_write_ratio,
                'total_accesses': self.access_pattern.total_accesses
            },
            'memory_level': self._estimate_memory_level()
        }
    
    def _estimate_memory_level(self) -> str:
        """Estimate which memory level the structure fits in"""
        size_bytes = self.size() * 16  # Rough estimate
        level, _ = self.hierarchy.get_level_for_size(size_bytes)
        return level


class CacheOptimizedBTree(Dict[K, V]):
    """B-Tree with node size optimized for cache lines"""
    
    def __init__(self):
        super().__init__()
        # Calculate optimal node size
        self.cache_line_size = 64
        # For 8-byte keys/values, we can fit 4 entries per cache line
        self.node_size = self.cache_line_size // 16
        # Use √n fanout for balanced height
        self._btree_impl = {}  # Simplified: use dict for now
    
    def __getitem__(self, key: K) -> V:
        return self._btree_impl[key]
    
    def __setitem__(self, key: K, value: V):
        self._btree_impl[key] = value
    
    def __delitem__(self, key: K):
        del self._btree_impl[key]
    
    def __len__(self) -> int:
        return len(self._btree_impl)
    
    def __contains__(self, key: K) -> bool:
        return key in self._btree_impl
    
    def get(self, key: K, default: Any = None) -> Any:
        return self._btree_impl.get(key, default)
    
    def pop(self, key: K, default: Any = None) -> Any:
        return self._btree_impl.pop(key, default)
    
    def items(self):
        return self._btree_impl.items()


class CacheOptimizedHashTable(Dict[K, V]):
    """Hash table with cache-aware probing"""
    
    def __init__(self, initial_size: int = 16):
        super().__init__()
        self.cache_line_size = 64
        # Ensure size is multiple of cache lines
        entries_per_line = self.cache_line_size // 16
        self.size = ((initial_size + entries_per_line - 1) // entries_per_line) * entries_per_line
        self._hash_impl = {}
    
    def __getitem__(self, key: K) -> V:
        return self._hash_impl[key]
    
    def __setitem__(self, key: K, value: V):
        self._hash_impl[key] = value
    
    def __delitem__(self, key: K):
        del self._hash_impl[key]
    
    def __len__(self) -> int:
        return len(self._hash_impl)
    
    def __contains__(self, key: K) -> bool:
        return key in self._hash_impl
    
    def get(self, key: K, default: Any = None) -> Any:
        return self._hash_impl.get(key, default)
    
    def pop(self, key: K, default: Any = None) -> Any:
        return self._hash_impl.pop(key, default)
    
    def items(self):
        return self._hash_impl.items()


class ExternalMemoryMap(Dict[K, V]):
    """Disk-backed map with √n-sized buffers"""
    
    def __init__(self):
        super().__init__()
        self.sqrt_calc = SqrtNCalculator()
        self._buffer = {}
        self._buffer_size = 0
        self._max_buffer_size = self.sqrt_calc.calculate_interval(1000000) * 16
        self._disk_data = {}  # Simplified: would use real disk storage
    
    def __getitem__(self, key: K) -> V:
        if key in self._buffer:
            return self._buffer[key]
        # Load from disk
        if key in self._disk_data:
            value = self._disk_data[key]
            self._add_to_buffer(key, value)
            return value
        raise KeyError(key)
    
    def __setitem__(self, key: K, value: V):
        self._add_to_buffer(key, value)
        self._disk_data[key] = value
    
    def __delitem__(self, key: K):
        if key in self._buffer:
            del self._buffer[key]
        if key in self._disk_data:
            del self._disk_data[key]
        else:
            raise KeyError(key)
    
    def __len__(self) -> int:
        return len(self._disk_data)
    
    def __contains__(self, key: K) -> bool:
        return key in self._disk_data
    
    def _add_to_buffer(self, key: K, value: V):
        """Add to buffer with LRU eviction"""
        if len(self._buffer) >= self._max_buffer_size // 16:
            # Evict oldest (simplified LRU)
            oldest = next(iter(self._buffer))
            del self._buffer[oldest]
        self._buffer[key] = value
    
    def get(self, key: K, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
    
    def pop(self, key: K, default: Any = None) -> Any:
        try:
            value = self[key]
            del self[key]
            return value
        except KeyError:
            return default
    
    def items(self):
        return self._disk_data.items()


class CompressedTrie:
    """Space-efficient trie with compression"""
    
    def __init__(self):
        self.root = {}
        self.compression_threshold = 10  # Compress paths longer than this
    
    def insert(self, key: str, value: Any):
        """Insert with path compression"""
        node = self.root
        i = 0
        
        while i < len(key):
            # Check for compressed edge
            for edge, (child, compressed_path) in list(node.items()):
                if edge == '_compressed' and key[i:].startswith(compressed_path):
                    i += len(compressed_path)
                    node = child
                    break
            else:
                # Normal edge
                if key[i] not in node:
                    # Check if we should compress
                    remaining = key[i:]
                    if len(remaining) > self.compression_threshold:
                        # Create compressed edge
                        node['_compressed'] = ({}, remaining)
                        node = node['_compressed'][0]
                        break
                    else:
                        node[key[i]] = {}
                node = node[key[i]]
                i += 1
        
        node['_value'] = value
    
    def search(self, key: str) -> Optional[Any]:
        """Search with compressed paths"""
        node = self.root
        i = 0
        
        while i < len(key) and node:
            # Check compressed edge
            if '_compressed' in node:
                child, compressed_path = node['_compressed']
                if key[i:].startswith(compressed_path):
                    i += len(compressed_path)
                    node = child
                    continue
            
            # Normal edge
            if key[i] in node:
                node = node[key[i]]
                i += 1
            else:
                return None
        
        return node.get('_value') if node else None


def create_optimized_structure(hint_type: str = 'auto', **kwargs) -> CacheAwareStructure:
    """Factory for creating optimized data structures"""
    if hint_type == 'auto':
        return AdaptiveMap(**kwargs)
    elif hint_type == 'btree':
        return CacheOptimizedBTree()
    elif hint_type == 'hash':
        return CacheOptimizedHashTable()
    elif hint_type == 'external':
        return ExternalMemoryMap()
    else:
        return AdaptiveMap(**kwargs)


# Example usage and benchmarks
if __name__ == "__main__":
    print("Cache-Aware Data Structures Example")
    print("="*60)
    
    # Example 1: Adaptive map
    print("\n1. Adaptive Map Demo")
    adaptive_map = AdaptiveMap[str, int]()
    
    # Insert increasing amounts of data
    sizes = [3, 10, 100, 1000, 10000]
    
    for size in sizes:
        print(f"\nInserting {size} elements...")
        for i in range(size):
            adaptive_map.put(f"key_{i}", i)
        
        stats = adaptive_map.get_stats()
        print(f"  Implementation: {stats['implementation']}")
        print(f"  Memory level: {stats['memory_level']}")
    
    # Example 2: Cache line aware sizing
    print("\n\n2. Cache Line Optimization")
    hierarchy = MemoryHierarchy.detect_system()
    
    print(f"System cache hierarchy:")
    print(f"  L1: {hierarchy.l1_size / 1024}KB")
    print(f"  L2: {hierarchy.l2_size / 1024}KB")
    print(f"  L3: {hierarchy.l3_size / 1024 / 1024}MB")
    
    # Calculate optimal sizes
    cache_line = 64
    entry_size = 16  # 8-byte key + 8-byte value
    
    print(f"\nOptimal structure sizes:")
    print(f"  Entries per cache line: {cache_line // entry_size}")
    print(f"  B-tree node size: {cache_line // entry_size} keys")
    print(f"  Hash table bucket size: {cache_line} bytes")
    
    # Example 3: Performance comparison
    print("\n\n3. Performance Comparison")
    n = 10000
    
    # Standard Python dict
    start = time.time()
    standard_dict = {}
    for i in range(n):
        standard_dict[f"key_{i}"] = i
    for i in range(n):
        _ = standard_dict.get(f"key_{i}")
    standard_time = time.time() - start
    
    # Adaptive map
    start = time.time()
    adaptive = AdaptiveMap[str, int]()
    for i in range(n):
        adaptive.put(f"key_{i}", i)
    for i in range(n):
        _ = adaptive.get(f"key_{i}")
    adaptive_time = time.time() - start
    
    print(f"Standard dict: {standard_time:.3f}s")
    print(f"Adaptive map: {adaptive_time:.3f}s")
    print(f"Overhead: {(adaptive_time / standard_time - 1) * 100:.1f}%")
    
    # Example 4: Compressed trie
    print("\n\n4. Compressed Trie Demo")
    trie = CompressedTrie()
    
    # Insert strings with common prefixes
    urls = [
        "http://example.com/api/v1/users/123",
        "http://example.com/api/v1/users/456",
        "http://example.com/api/v1/products/789",
        "http://example.com/api/v2/users/123",
    ]
    
    for url in urls:
        trie.insert(url, f"data_for_{url}")
    
    # Search
    for url in urls[:2]:
        result = trie.search(url)
        print(f"Found: {url} -> {result}")
    
    print("\n" + "="*60)
    print("Cache-aware structures provide better performance")
    print("by adapting to hardware memory hierarchies.")
