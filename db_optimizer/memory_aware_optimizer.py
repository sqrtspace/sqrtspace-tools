#!/usr/bin/env python3
"""
Memory-Aware Query Optimizer: Database query optimizer considering memory hierarchies

Features:
- Cost Model: Include L3/RAM/SSD boundaries in cost calculations
- Algorithm Selection: Choose between hash/sort/nested-loop based on true costs
- Buffer Sizing: Automatically size buffers to √(data_size)
- Spill Planning: Optimize when and how to spill to disk
- AI Explanations: Clear reasoning for optimization decisions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import psutil
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import re
import tempfile
from pathlib import Path

# Import core components
from core.spacetime_core import (
    MemoryHierarchy,
    SqrtNCalculator,
    OptimizationStrategy,
    StrategyAnalyzer
)


class JoinAlgorithm(Enum):
    """Join algorithms with different space-time tradeoffs"""
    NESTED_LOOP = "nested_loop"      # O(1) space, O(n*m) time
    SORT_MERGE = "sort_merge"        # O(n+m) space, O(n log n + m log m) time
    HASH_JOIN = "hash_join"          # O(min(n,m)) space, O(n+m) time
    BLOCK_NESTED = "block_nested"    # O(√n) space, O(n*m/√n) time


class ScanType(Enum):
    """Scan types for table access"""
    SEQUENTIAL = "sequential"        # Full table scan
    INDEX = "index"                  # Index scan
    BITMAP = "bitmap"                # Bitmap index scan


@dataclass
class TableStats:
    """Statistics about a database table"""
    name: str
    row_count: int
    avg_row_size: int
    total_size: int
    indexes: List[str]
    cardinality: Dict[str, int]  # Column -> distinct values


@dataclass
class QueryNode:
    """Node in query execution plan"""
    operation: str
    algorithm: Optional[str]
    estimated_rows: int
    estimated_size: int
    estimated_cost: float
    memory_required: int
    memory_level: str
    children: List['QueryNode']
    explanation: str


@dataclass
class OptimizationResult:
    """Result of query optimization"""
    original_plan: QueryNode
    optimized_plan: QueryNode
    memory_saved: int
    estimated_speedup: float
    buffer_sizes: Dict[str, int]
    spill_strategy: Dict[str, str]
    explanation: str


class CostModel:
    """Cost model considering memory hierarchy"""
    
    def __init__(self, hierarchy: MemoryHierarchy):
        self.hierarchy = hierarchy
        
        # Cost factors (relative to L1 access)
        self.cpu_factor = 0.1
        self.l1_factor = 1.0
        self.l2_factor = 4.0
        self.l3_factor = 12.0
        self.ram_factor = 100.0
        self.disk_factor = 10000.0
    
    def calculate_scan_cost(self, table_size: int, scan_type: ScanType) -> float:
        """Calculate cost of scanning a table"""
        level, latency = self.hierarchy.get_level_for_size(table_size)
        
        if scan_type == ScanType.SEQUENTIAL:
            # Sequential scan benefits from prefetching
            return table_size * latency * 0.5
        elif scan_type == ScanType.INDEX:
            # Random access pattern
            return table_size * latency * 2.0
        else:  # BITMAP
            # Mixed pattern
            return table_size * latency
    
    def calculate_join_cost(self, left_size: int, right_size: int, 
                          algorithm: JoinAlgorithm, buffer_size: int) -> float:
        """Calculate cost of join operation"""
        if algorithm == JoinAlgorithm.NESTED_LOOP:
            # O(n*m) comparisons, minimal memory
            comparisons = left_size * right_size
            memory_used = buffer_size
            
        elif algorithm == JoinAlgorithm.SORT_MERGE:
            # Sort both sides then merge
            sort_cost = left_size * np.log2(left_size) + right_size * np.log2(right_size)
            merge_cost = left_size + right_size
            comparisons = sort_cost + merge_cost
            memory_used = left_size + right_size
            
        elif algorithm == JoinAlgorithm.HASH_JOIN:
            # Build hash table on smaller side
            build_size = min(left_size, right_size)
            probe_size = max(left_size, right_size)
            comparisons = build_size + probe_size
            memory_used = build_size * 1.5  # Hash table overhead
            
        else:  # BLOCK_NESTED
            # Process in √n blocks
            block_size = int(np.sqrt(min(left_size, right_size)))
            blocks = (left_size // block_size) * (right_size // block_size)
            comparisons = blocks * block_size * block_size
            memory_used = block_size
        
        # Get memory level for this operation
        level, latency = self.hierarchy.get_level_for_size(memory_used)
        
        # Add spill cost if memory exceeded
        spill_cost = 0
        if memory_used > buffer_size:
            spill_ratio = memory_used / buffer_size
            spill_cost = comparisons * self.disk_factor * 0.1 * spill_ratio
        
        return comparisons * latency + spill_cost
    
    def calculate_sort_cost(self, data_size: int, memory_limit: int) -> float:
        """Calculate cost of sorting with limited memory"""
        if data_size <= memory_limit:
            # In-memory sort
            comparisons = data_size * np.log2(data_size)
            level, latency = self.hierarchy.get_level_for_size(data_size)
            return comparisons * latency
        else:
            # External sort with √n memory
            runs = data_size // memory_limit
            merge_passes = np.log2(runs)
            total_io = data_size * merge_passes * 2  # Read + write
            return total_io * self.disk_factor


class QueryAnalyzer:
    """Analyze queries and extract operations"""
    
    @staticmethod
    def parse_query(sql: str) -> Dict[str, Any]:
        """Parse SQL query to extract operations"""
        sql_upper = sql.upper()
        
        # Extract tables
        tables = []
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            tables.append(from_match.group(1))
        
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        tables.extend(join_matches)
        
        # Extract join conditions
        joins = []
        join_pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        for match in re.finditer(join_pattern, sql, re.IGNORECASE):
            joins.append({
                'left_table': match.group(1),
                'left_col': match.group(2),
                'right_table': match.group(3),
                'right_col': match.group(4)
            })
        
        # Extract filters
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql_upper)
        filters = where_match.group(1) if where_match else None
        
        # Extract aggregations
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
        aggregations = []
        for func in agg_functions:
            if func in sql_upper:
                aggregations.append(func)
        
        # Extract order by
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:LIMIT|$)', sql_upper)
        order_by = order_match.group(1) if order_match else None
        
        return {
            'tables': tables,
            'joins': joins,
            'filters': filters,
            'aggregations': aggregations,
            'order_by': order_by
        }


class MemoryAwareOptimizer:
    """Main query optimizer with memory awareness"""
    
    def __init__(self, connection: sqlite3.Connection, 
                 memory_limit: Optional[int] = None):
        self.conn = connection
        self.hierarchy = MemoryHierarchy.detect_system()
        self.cost_model = CostModel(self.hierarchy)
        self.memory_limit = memory_limit or int(psutil.virtual_memory().available * 0.5)
        self.table_stats = {}
        
        # Collect table statistics
        self._collect_statistics()
    
    def _collect_statistics(self):
        """Collect statistics about database tables"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            # Estimate row size (simplified)
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            avg_row_size = len(columns) * 20  # Rough estimate
            
            # Get indexes
            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = [idx[1] for idx in cursor.fetchall()]
            
            self.table_stats[table_name] = TableStats(
                name=table_name,
                row_count=row_count,
                avg_row_size=avg_row_size,
                total_size=row_count * avg_row_size,
                indexes=indexes,
                cardinality={}
            )
    
    def optimize_query(self, sql: str) -> OptimizationResult:
        """Optimize a SQL query considering memory constraints"""
        # Parse query
        query_info = QueryAnalyzer.parse_query(sql)
        
        # Build original plan
        original_plan = self._build_execution_plan(query_info, optimize=False)
        
        # Build optimized plan
        optimized_plan = self._build_execution_plan(query_info, optimize=True)
        
        # Calculate buffer sizes
        buffer_sizes = self._calculate_buffer_sizes(optimized_plan)
        
        # Determine spill strategy
        spill_strategy = self._determine_spill_strategy(optimized_plan)
        
        # Calculate improvements
        memory_saved = original_plan.memory_required - optimized_plan.memory_required
        estimated_speedup = original_plan.estimated_cost / optimized_plan.estimated_cost
        
        # Generate explanation
        explanation = self._generate_optimization_explanation(
            original_plan, optimized_plan, buffer_sizes
        )
        
        return OptimizationResult(
            original_plan=original_plan,
            optimized_plan=optimized_plan,
            memory_saved=memory_saved,
            estimated_speedup=estimated_speedup,
            buffer_sizes=buffer_sizes,
            spill_strategy=spill_strategy,
            explanation=explanation
        )
    
    def _build_execution_plan(self, query_info: Dict[str, Any], 
                            optimize: bool) -> QueryNode:
        """Build query execution plan"""
        tables = query_info['tables']
        joins = query_info['joins']
        
        if not tables:
            return QueryNode(
                operation="EMPTY",
                algorithm=None,
                estimated_rows=0,
                estimated_size=0,
                estimated_cost=0,
                memory_required=0,
                memory_level="L1",
                children=[],
                explanation="Empty query"
            )
        
        # Start with first table
        plan = self._create_scan_node(tables[0], query_info.get('filters'))
        
        # Add joins
        for i, join in enumerate(joins):
            if i + 1 < len(tables):
                right_table = tables[i + 1]
                right_scan = self._create_scan_node(right_table, None)
                
                # Choose join algorithm
                if optimize:
                    algorithm = self._choose_join_algorithm(
                        plan.estimated_size,
                        right_scan.estimated_size
                    )
                else:
                    algorithm = JoinAlgorithm.NESTED_LOOP
                
                plan = self._create_join_node(plan, right_scan, algorithm, join)
        
        # Add sort if needed
        if query_info.get('order_by'):
            plan = self._create_sort_node(plan, optimize)
        
        # Add aggregation if needed
        if query_info.get('aggregations'):
            plan = self._create_aggregation_node(plan, query_info['aggregations'])
        
        return plan
    
    def _create_scan_node(self, table_name: str, filters: Optional[str]) -> QueryNode:
        """Create table scan node"""
        stats = self.table_stats.get(table_name, TableStats(
            name=table_name,
            row_count=1000,
            avg_row_size=100,
            total_size=100000,
            indexes=[],
            cardinality={}
        ))
        
        # Estimate selectivity
        selectivity = 0.1 if filters else 1.0
        estimated_rows = int(stats.row_count * selectivity)
        estimated_size = estimated_rows * stats.avg_row_size
        
        # Choose scan type
        scan_type = ScanType.INDEX if stats.indexes and filters else ScanType.SEQUENTIAL
        
        # Calculate cost
        cost = self.cost_model.calculate_scan_cost(estimated_size, scan_type)
        
        level, _ = self.hierarchy.get_level_for_size(estimated_size)
        
        return QueryNode(
            operation=f"SCAN {table_name}",
            algorithm=scan_type.value,
            estimated_rows=estimated_rows,
            estimated_size=estimated_size,
            estimated_cost=cost,
            memory_required=estimated_size,
            memory_level=level,
            children=[],
            explanation=f"{scan_type.value} scan on {table_name}"
        )
    
    def _create_join_node(self, left: QueryNode, right: QueryNode,
                         algorithm: JoinAlgorithm, join_info: Dict) -> QueryNode:
        """Create join node"""
        # Estimate join output size
        join_selectivity = 0.1  # Simplified
        estimated_rows = int(left.estimated_rows * right.estimated_rows * join_selectivity)
        estimated_size = estimated_rows * (left.estimated_size // left.estimated_rows + 
                                         right.estimated_size // right.estimated_rows)
        
        # Calculate memory required
        if algorithm == JoinAlgorithm.HASH_JOIN:
            memory_required = min(left.estimated_size, right.estimated_size) * 1.5
        elif algorithm == JoinAlgorithm.SORT_MERGE:
            memory_required = left.estimated_size + right.estimated_size
        elif algorithm == JoinAlgorithm.BLOCK_NESTED:
            memory_required = int(np.sqrt(min(left.estimated_size, right.estimated_size)))
        else:  # NESTED_LOOP
            memory_required = 1000  # Minimal buffer
        
        # Calculate buffer size considering memory limit
        buffer_size = min(memory_required, self.memory_limit)
        
        # Calculate cost
        cost = self.cost_model.calculate_join_cost(
            left.estimated_rows, right.estimated_rows, algorithm, buffer_size
        )
        
        level, _ = self.hierarchy.get_level_for_size(memory_required)
        
        return QueryNode(
            operation="JOIN",
            algorithm=algorithm.value,
            estimated_rows=estimated_rows,
            estimated_size=estimated_size,
            estimated_cost=cost + left.estimated_cost + right.estimated_cost,
            memory_required=memory_required,
            memory_level=level,
            children=[left, right],
            explanation=f"{algorithm.value} join with {buffer_size / 1024:.0f}KB buffer"
        )
    
    def _create_sort_node(self, child: QueryNode, optimize: bool) -> QueryNode:
        """Create sort node"""
        if optimize:
            # Use √n memory for external sort
            memory_limit = int(np.sqrt(child.estimated_size))
        else:
            # Try to sort in memory
            memory_limit = child.estimated_size
        
        cost = self.cost_model.calculate_sort_cost(child.estimated_size, memory_limit)
        level, _ = self.hierarchy.get_level_for_size(memory_limit)
        
        return QueryNode(
            operation="SORT",
            algorithm="external_sort" if memory_limit < child.estimated_size else "quicksort",
            estimated_rows=child.estimated_rows,
            estimated_size=child.estimated_size,
            estimated_cost=cost + child.estimated_cost,
            memory_required=memory_limit,
            memory_level=level,
            children=[child],
            explanation=f"Sort with {memory_limit / 1024:.0f}KB memory"
        )
    
    def _create_aggregation_node(self, child: QueryNode, 
                               aggregations: List[str]) -> QueryNode:
        """Create aggregation node"""
        # Estimate groups (simplified)
        estimated_groups = int(np.sqrt(child.estimated_rows))
        estimated_size = estimated_groups * 100  # Rough estimate
        
        # Hash-based aggregation
        memory_required = estimated_size * 1.5
        
        level, _ = self.hierarchy.get_level_for_size(memory_required)
        
        return QueryNode(
            operation="AGGREGATE",
            algorithm="hash_aggregate",
            estimated_rows=estimated_groups,
            estimated_size=estimated_size,
            estimated_cost=child.estimated_cost + child.estimated_rows,
            memory_required=memory_required,
            memory_level=level,
            children=[child],
            explanation=f"Hash aggregation: {', '.join(aggregations)}"
        )
    
    def _choose_join_algorithm(self, left_size: int, right_size: int) -> JoinAlgorithm:
        """Choose optimal join algorithm based on sizes and memory"""
        min_size = min(left_size, right_size)
        max_size = max(left_size, right_size)
        
        # Can we fit hash table in memory?
        hash_memory = min_size * 1.5
        if hash_memory <= self.memory_limit:
            return JoinAlgorithm.HASH_JOIN
        
        # Can we fit both relations for sort-merge?
        sort_memory = left_size + right_size
        if sort_memory <= self.memory_limit:
            return JoinAlgorithm.SORT_MERGE
        
        # Use block nested loop with √n memory
        sqrt_memory = int(np.sqrt(min_size))
        if sqrt_memory <= self.memory_limit:
            return JoinAlgorithm.BLOCK_NESTED
        
        # Fall back to nested loop
        return JoinAlgorithm.NESTED_LOOP
    
    def _calculate_buffer_sizes(self, plan: QueryNode) -> Dict[str, int]:
        """Calculate optimal buffer sizes for operations"""
        buffer_sizes = {}
        
        def traverse(node: QueryNode, path: str = ""):
            if node.operation == "SCAN":
                # √n buffer for sequential scans
                buffer_size = min(
                    int(np.sqrt(node.estimated_size)),
                    self.memory_limit // 10
                )
                buffer_sizes[f"{path}scan_buffer"] = buffer_size
            
            elif node.operation == "JOIN":
                # Optimal buffer based on algorithm
                if node.algorithm == "block_nested":
                    buffer_size = int(np.sqrt(node.memory_required))
                else:
                    buffer_size = min(node.memory_required, self.memory_limit // 4)
                buffer_sizes[f"{path}join_buffer"] = buffer_size
            
            elif node.operation == "SORT":
                # √n buffer for external sort
                buffer_size = int(np.sqrt(node.estimated_size))
                buffer_sizes[f"{path}sort_buffer"] = buffer_size
            
            for i, child in enumerate(node.children):
                traverse(child, f"{path}{node.operation}_{i}_")
        
        traverse(plan)
        return buffer_sizes
    
    def _determine_spill_strategy(self, plan: QueryNode) -> Dict[str, str]:
        """Determine when and how to spill to disk"""
        spill_strategy = {}
        
        def traverse(node: QueryNode, path: str = ""):
            if node.memory_required > self.memory_limit:
                if node.operation == "JOIN":
                    if node.algorithm == "hash_join":
                        spill_strategy[path] = "grace_hash_join"
                    elif node.algorithm == "sort_merge":
                        spill_strategy[path] = "external_sort_both_inputs"
                    else:
                        spill_strategy[path] = "block_nested_with_spill"
                
                elif node.operation == "SORT":
                    spill_strategy[path] = "multi_pass_external_sort"
                
                elif node.operation == "AGGREGATE":
                    spill_strategy[path] = "spill_partial_aggregates"
            
            for i, child in enumerate(node.children):
                traverse(child, f"{path}{node.operation}_{i}_")
        
        traverse(plan)
        return spill_strategy
    
    def _generate_optimization_explanation(self, original: QueryNode,
                                         optimized: QueryNode,
                                         buffer_sizes: Dict[str, int]) -> str:
        """Generate AI-style explanation of optimizations"""
        explanations = []
        
        # Overall improvement
        memory_reduction = (1 - optimized.memory_required / original.memory_required) * 100
        speedup = original.estimated_cost / optimized.estimated_cost
        
        explanations.append(
            f"Optimized query plan reduces memory usage by {memory_reduction:.1f}% "
            f"with {speedup:.1f}x estimated speedup."
        )
        
        # Specific optimizations
        def compare_nodes(orig: QueryNode, opt: QueryNode, path: str = ""):
            if orig.algorithm != opt.algorithm:
                if orig.operation == "JOIN":
                    explanations.append(
                        f"Changed {path} from {orig.algorithm} to {opt.algorithm} "
                        f"saving {(orig.memory_required - opt.memory_required) / 1024:.0f}KB"
                    )
                elif orig.operation == "SORT":
                    explanations.append(
                        f"Using external sort at {path} with √n memory "
                        f"({opt.memory_required / 1024:.0f}KB instead of "
                        f"{orig.memory_required / 1024:.0f}KB)"
                    )
            
            for i, (orig_child, opt_child) in enumerate(zip(orig.children, opt.children)):
                compare_nodes(orig_child, opt_child, f"{path}{orig.operation}_{i}_")
        
        compare_nodes(original, optimized)
        
        # Buffer recommendations
        total_buffers = sum(buffer_sizes.values())
        explanations.append(
            f"Allocated {len(buffer_sizes)} buffers totaling "
            f"{total_buffers / 1024:.0f}KB for optimal performance."
        )
        
        # Memory hierarchy awareness
        if optimized.memory_level != original.memory_level:
            explanations.append(
                f"Optimized plan fits in {optimized.memory_level} "
                f"instead of {original.memory_level}, reducing latency."
            )
        
        return " ".join(explanations)
    
    def explain_plan(self, plan: QueryNode, indent: int = 0) -> str:
        """Generate text representation of query plan"""
        lines = []
        prefix = "  " * indent
        
        lines.append(f"{prefix}{plan.operation} ({plan.algorithm})")
        lines.append(f"{prefix}  Rows: {plan.estimated_rows:,}")
        lines.append(f"{prefix}  Size: {plan.estimated_size / 1024:.1f}KB")
        lines.append(f"{prefix}  Memory: {plan.memory_required / 1024:.1f}KB ({plan.memory_level})")
        lines.append(f"{prefix}  Cost: {plan.estimated_cost:.0f}")
        
        for child in plan.children:
            lines.append(self.explain_plan(child, indent + 1))
        
        return "\n".join(lines)
    
    def apply_hints(self, sql: str, target: str = 'latency', 
                   memory_limit: Optional[str] = None) -> str:
        """Apply optimizer hints to SQL query"""
        # Parse memory limit if provided
        if memory_limit:
            limit_match = re.match(r'(\d+)(MB|GB)?', memory_limit, re.IGNORECASE)
            if limit_match:
                value = int(limit_match.group(1))
                unit = limit_match.group(2) or 'MB'
                if unit.upper() == 'GB':
                    value *= 1024
                self.memory_limit = value * 1024 * 1024
        
        # Optimize query
        result = self.optimize_query(sql)
        
        # Generate hint comment
        hint = f"/* SpaceTime Optimizer: {result.explanation} */\n"
        
        return hint + sql


# Example usage and testing
if __name__ == "__main__":
    # Create test database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create test tables
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            amount REAL,
            date TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            price REAL
        )
    """)
    
    # Insert test data
    for i in range(10000):
        cursor.execute("INSERT INTO customers VALUES (?, ?, ?)",
                      (i, f"Customer {i}", f"Country {i % 100}"))
    
    for i in range(50000):
        cursor.execute("INSERT INTO orders VALUES (?, ?, ?, ?)",
                      (i, i % 10000, i * 10.0, '2024-01-01'))
    
    for i in range(1000):
        cursor.execute("INSERT INTO products VALUES (?, ?, ?)",
                      (i, f"Product {i}", i * 5.0))
    
    conn.commit()
    
    # Create optimizer
    optimizer = MemoryAwareOptimizer(conn, memory_limit=1024*1024)  # 1MB limit
    
    # Test queries
    queries = [
        """
        SELECT c.name, SUM(o.amount)
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE c.country = 'Country 1'
        GROUP BY c.name
        ORDER BY SUM(o.amount) DESC
        """,
        
        """
        SELECT *
        FROM orders o1
        JOIN orders o2 ON o1.customer_id = o2.customer_id
        WHERE o1.amount > 1000
        """
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}:")
        print(query.strip())
        print("="*60)
        
        # Optimize query
        result = optimizer.optimize_query(query)
        
        print("\nOriginal Plan:")
        print(optimizer.explain_plan(result.original_plan))
        
        print("\nOptimized Plan:")
        print(optimizer.explain_plan(result.optimized_plan))
        
        print(f"\nOptimization Results:")
        print(f"  Memory Saved: {result.memory_saved / 1024:.1f}KB")
        print(f"  Estimated Speedup: {result.estimated_speedup:.1f}x")
        print(f"\nBuffer Sizes:")
        for name, size in result.buffer_sizes.items():
            print(f"  {name}: {size / 1024:.1f}KB")
        
        if result.spill_strategy:
            print(f"\nSpill Strategy:")
            for op, strategy in result.spill_strategy.items():
                print(f"  {op}: {strategy}")
        
        print(f"\nExplanation: {result.explanation}")
    
    # Test hint application
    print("\n" + "="*60)
    print("Query with hints:")
    print("="*60)
    
    hinted_sql = optimizer.apply_hints(
        "SELECT * FROM customers c JOIN orders o ON c.id = o.customer_id",
        target='memory',
        memory_limit='512KB'
    )
    print(hinted_sql)
    
    conn.close()
