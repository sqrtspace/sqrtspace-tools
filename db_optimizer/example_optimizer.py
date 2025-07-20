#!/usr/bin/env python3
"""
Example demonstrating Memory-Aware Query Optimizer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_aware_optimizer import MemoryAwareOptimizer
import sqlite3
import time


def create_test_database():
    """Create a test database with sample data"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            email TEXT,
            created_at TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE comments (
            id INTEGER PRIMARY KEY,
            post_id INTEGER,
            user_id INTEGER,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (post_id) REFERENCES posts(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Insert sample data
    print("Creating test data...")
    
    # Users
    for i in range(1000):
        cursor.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            (i, f"user{i}", f"user{i}@example.com", "2024-01-01")
        )
    
    # Posts
    for i in range(5000):
        cursor.execute(
            "INSERT INTO posts VALUES (?, ?, ?, ?, ?)",
            (i, i % 1000, f"Post {i}", f"Content for post {i}", "2024-01-02")
        )
    
    # Comments
    for i in range(20000):
        cursor.execute(
            "INSERT INTO comments VALUES (?, ?, ?, ?, ?)",
            (i, i % 5000, i % 1000, f"Comment {i}", "2024-01-03")
        )
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_posts_user ON posts(user_id)")
    cursor.execute("CREATE INDEX idx_comments_post ON comments(post_id)")
    cursor.execute("CREATE INDEX idx_comments_user ON comments(user_id)")
    
    conn.commit()
    return conn


def demonstrate_optimizer(conn):
    """Demonstrate query optimization capabilities"""
    # Create optimizer with 2MB memory limit
    optimizer = MemoryAwareOptimizer(conn, memory_limit=2*1024*1024)
    
    print("\n" + "="*60)
    print("Memory-Aware Query Optimizer Demonstration")
    print("="*60)
    
    # Example 1: Simple join query
    query1 = """
        SELECT u.username, COUNT(p.id) as post_count
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
        GROUP BY u.username
        ORDER BY post_count DESC
        LIMIT 10
    """
    
    print("\nExample 1: User post counts")
    print("-" * 40)
    result1 = optimizer.optimize_query(query1)
    
    print("Memory saved:", f"{result1.memory_saved / 1024:.1f}KB")
    print("Speedup:", f"{result1.estimated_speedup:.1f}x")
    print("\nOptimization:", result1.explanation)
    
    # Example 2: Complex multi-join
    query2 = """
        SELECT p.title, COUNT(c.id) as comment_count
        FROM posts p
        JOIN comments c ON p.id = c.post_id
        JOIN users u ON p.user_id = u.id
        WHERE u.created_at > '2023-12-01'
        GROUP BY p.title
        ORDER BY comment_count DESC
    """
    
    print("\n\nExample 2: Posts with most comments")
    print("-" * 40)
    result2 = optimizer.optimize_query(query2)
    
    print("Original memory:", f"{result2.original_plan.memory_required / 1024:.1f}KB")
    print("Optimized memory:", f"{result2.optimized_plan.memory_required / 1024:.1f}KB")
    print("Speedup:", f"{result2.estimated_speedup:.1f}x")
    
    # Show buffer allocation
    print("\nBuffer allocation:")
    for buffer_name, size in result2.buffer_sizes.items():
        print(f"  {buffer_name}: {size / 1024:.1f}KB")
    
    # Example 3: Self-join (typically memory intensive)
    query3 = """
        SELECT u1.username, u2.username
        FROM users u1
        JOIN users u2 ON u1.id < u2.id
        WHERE u1.email LIKE '%@gmail.com'
        AND u2.email LIKE '%@gmail.com'
        LIMIT 100
    """
    
    print("\n\nExample 3: Self-join optimization")
    print("-" * 40)
    result3 = optimizer.optimize_query(query3)
    
    print("Join algorithm chosen:", result3.optimized_plan.children[0].algorithm if result3.optimized_plan.children else "N/A")
    print("Memory level:", result3.optimized_plan.memory_level)
    print("\nOptimization:", result3.explanation)
    
    # Show actual execution comparison
    print("\n\nActual Execution Comparison")
    print("-" * 40)
    
    # Execute with standard SQLite
    start = time.time()
    cursor = conn.cursor()
    cursor.execute("PRAGMA cache_size = -2000")  # 2MB cache
    cursor.execute(query1)
    _ = cursor.fetchall()
    standard_time = time.time() - start
    
    # Execute with optimized settings
    start = time.time()
    # Apply √n cache size
    optimal_cache = int((1000 * 5000) ** 0.5) // 1024  # √(users * posts) in KB
    cursor.execute(f"PRAGMA cache_size = -{optimal_cache}")
    cursor.execute(query1)
    _ = cursor.fetchall()
    optimized_time = time.time() - start
    
    print(f"Standard execution: {standard_time:.3f}s")
    print(f"Optimized execution: {optimized_time:.3f}s")
    print(f"Actual speedup: {standard_time / optimized_time:.1f}x")


def show_query_plans(conn):
    """Show visual representation of query plans"""
    optimizer = MemoryAwareOptimizer(conn, memory_limit=1024*1024)  # 1MB limit
    
    print("\n\nQuery Plan Visualization")
    print("="*60)
    
    query = """
        SELECT u.username, COUNT(c.id) as activity
        FROM users u
        JOIN posts p ON u.id = p.user_id
        JOIN comments c ON p.id = c.post_id
        GROUP BY u.username
        ORDER BY activity DESC
    """
    
    result = optimizer.optimize_query(query)
    
    print("\nOriginal Plan:")
    print(optimizer.explain_plan(result.original_plan))
    
    print("\n\nOptimized Plan:")
    print(optimizer.explain_plan(result.optimized_plan))
    
    # Show memory hierarchy utilization
    print("\n\nMemory Hierarchy Utilization:")
    print("-" * 40)
    
    def show_memory_usage(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}{node.operation}: {node.memory_level} "
              f"({node.memory_required / 1024:.1f}KB)")
        for child in node.children:
            show_memory_usage(child, indent + 1)
    
    show_memory_usage(result.optimized_plan)


def main():
    """Run demonstration"""
    # Create test database
    conn = create_test_database()
    
    # Run demonstrations
    demonstrate_optimizer(conn)
    show_query_plans(conn)
    
    # Show hint usage
    print("\n\nSQL with Optimizer Hints")
    print("="*60)
    
    optimizer = MemoryAwareOptimizer(conn, memory_limit=512*1024)  # 512KB limit
    
    original_sql = "SELECT * FROM users u JOIN posts p ON u.id = p.user_id"
    
    # Optimize for low memory
    memory_optimized = optimizer.apply_hints(original_sql, target='memory', memory_limit='256KB')
    print("\nMemory-optimized SQL:")
    print(memory_optimized)
    
    # Optimize for speed
    speed_optimized = optimizer.apply_hints(original_sql, target='latency')
    print("\nSpeed-optimized SQL:")
    print(speed_optimized)
    
    conn.close()
    
    print("\n" + "="*60)
    print("Demonstration complete!")
    print("="*60)


if __name__ == "__main__":
    main()