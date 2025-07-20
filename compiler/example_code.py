#!/usr/bin/env python3
"""
Example code to demonstrate SpaceTime Compiler optimizations
This file contains various patterns that can be optimized.
"""

import numpy as np
from typing import List, Dict, Tuple


def process_large_dataset(data: List[float], threshold: float) -> Dict[str, List[float]]:
    """Process large dataset with multiple optimization opportunities"""
    # Opportunity 1: Large list accumulation
    filtered_data = []
    for value in data:
        if value > threshold:
            filtered_data.append(value * 2.0)
    
    # Opportunity 2: Sorting large data
    sorted_data = sorted(filtered_data)
    
    # Opportunity 3: Accumulation in loop
    total = 0.0
    count = 0
    for value in sorted_data:
        total += value
        count += 1
    
    mean = total / count if count > 0 else 0.0
    
    # Opportunity 4: Large comprehension
    squared_deviations = [(x - mean) ** 2 for x in sorted_data]
    
    # Opportunity 5: Grouping with accumulation
    groups = {}
    for i, value in enumerate(sorted_data):
        group_key = f"group_{int(value // 100)}"
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(value)
    
    return groups


def matrix_computation(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Matrix operations that can benefit from cache blocking"""
    # Opportunity: Matrix multiplication
    result1 = np.dot(A, B)
    
    # Opportunity: Another matrix multiplication
    result2 = np.dot(result1, C)
    
    # Opportunity: Element-wise operations in loop
    n_rows, n_cols = result2.shape
    for i in range(n_rows):
        for j in range(n_cols):
            result2[i, j] = np.sqrt(result2[i, j]) if result2[i, j] > 0 else 0
    
    return result2


def analyze_log_files(log_paths: List[str]) -> Dict[str, int]:
    """Analyze multiple log files - external memory opportunity"""
    # Opportunity: Large accumulation
    all_entries = []
    for path in log_paths:
        with open(path, 'r') as f:
            entries = f.readlines()
            all_entries.extend(entries)
    
    # Opportunity: Processing large list
    error_counts = {}
    for entry in all_entries:
        if 'ERROR' in entry:
            error_type = extract_error_type(entry)
            if error_type not in error_counts:
                error_counts[error_type] = 0
            error_counts[error_type] += 1
    
    return error_counts


def extract_error_type(log_entry: str) -> str:
    """Helper function to extract error type"""
    # Simplified error extraction
    if 'FileNotFound' in log_entry:
        return 'FileNotFound'
    elif 'ValueError' in log_entry:
        return 'ValueError'
    elif 'KeyError' in log_entry:
        return 'KeyError'
    else:
        return 'Unknown'


def simulate_particles(n_particles: int, n_steps: int) -> List[np.ndarray]:
    """Particle simulation with checkpointing opportunity"""
    # Initialize particles
    positions = np.random.rand(n_particles, 3)
    velocities = np.random.rand(n_particles, 3) - 0.5
    
    # Opportunity: Large trajectory accumulation
    trajectory = []
    
    # Opportunity: Large loop with accumulation
    for step in range(n_steps):
        # Update positions
        positions += velocities * 0.01  # dt = 0.01
        
        # Apply boundary conditions
        positions = np.clip(positions, 0, 1)
        
        # Store position (checkpoint opportunity)
        trajectory.append(positions.copy())
        
        # Apply some forces
        velocities *= 0.99  # Damping
    
    return trajectory


def build_index(documents: List[str]) -> Dict[str, List[int]]:
    """Build inverted index - memory optimization opportunity"""
    # Opportunity: Large dictionary with lists
    index = {}
    
    # Opportunity: Nested loops with accumulation
    for doc_id, document in enumerate(documents):
        words = document.lower().split()
        
        for word in words:
            if word not in index:
                index[word] = []
            index[word].append(doc_id)
    
    # Opportunity: Sorting index values
    for word in index:
        index[word] = sorted(set(index[word]))
    
    return index


def process_stream(data_stream) -> Tuple[float, float]:
    """Process streaming data - generator opportunity"""
    # Opportunity: Could use generator instead of list
    values = [float(x) for x in data_stream]
    
    # Calculate statistics
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    
    return mean, variance


def graph_analysis(adjacency_list: Dict[int, List[int]], start_node: int) -> List[int]:
    """Graph traversal - memory-bounded opportunity"""
    visited = set()
    # Opportunity: Queue could be memory-bounded
    queue = [start_node]
    traversal_order = []
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            
            # Add all neighbors
            for neighbor in adjacency_list.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return traversal_order


if __name__ == "__main__":
    # Example usage
    print("This file demonstrates various optimization opportunities")
    print("Run the SpaceTime Compiler on this file to see optimizations")
    
    # Small examples
    data = list(range(10000))
    result = process_large_dataset(data, 5000)
    print(f"Processed {len(data)} items into {len(result)} groups")
    
    # Matrix example  
    A = np.random.rand(100, 100)
    B = np.random.rand(100, 100)
    C = np.random.rand(100, 100)
    result_matrix = matrix_computation(A, B, C)
    print(f"Matrix computation result shape: {result_matrix.shape}")