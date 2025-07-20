#!/usr/bin/env python3
"""
Example demonstrating SpaceTime Configuration Advisor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_advisor import (
    ConfigurationAdvisor,
    SystemType,
    WorkloadType
)
import json


def example_postgresql_tuning():
    """Tune PostgreSQL for different workloads"""
    print("="*60)
    print("PostgreSQL Tuning Example")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    # Scenario 1: E-commerce website (OLTP)
    print("\n1. E-commerce Website (OLTP)")
    print("-"*40)
    
    ecommerce_workload = {
        'read_ratio': 0.9,          # 90% reads
        'working_set_gb': 20,       # Hot data
        'total_data_gb': 200,       # Total database
        'qps': 5000,                # Queries per second
        'connections': 300,         # Concurrent users
        'latency_sla_ms': 50        # 50ms SLA
    }
    
    config = advisor.analyze(
        workload_data=ecommerce_workload,
        target=SystemType.DATABASE
    )
    
    print(f"Configuration: {config.explanation}")
    print("\nKey settings:")
    for k, v in config.settings.items():
        print(f"  {k} = {v}")
    
    # Scenario 2: Analytics warehouse (OLAP)
    print("\n\n2. Analytics Data Warehouse (OLAP)")
    print("-"*40)
    
    analytics_workload = {
        'read_ratio': 0.99,         # Almost all reads
        'working_set_gb': 500,      # Large working set
        'total_data_gb': 5000,      # 5TB warehouse
        'qps': 100,                 # Complex queries
        'connections': 50,          # Fewer concurrent users
        'analytics': True,          # Analytics flag
        'avg_request_kb': 1000      # Large results
    }
    
    config = advisor.analyze(
        workload_data=analytics_workload,
        target=SystemType.DATABASE
    )
    
    print(f"Configuration: {config.explanation}")
    print("\nKey settings:")
    for k, v in config.settings.items():
        print(f"  {k} = {v}")


def example_jvm_tuning():
    """Tune JVM for different applications"""
    print("\n\n" + "="*60)
    print("JVM Tuning Example")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    # Scenario 1: Low-latency trading system
    print("\n1. Low-Latency Trading System")
    print("-"*40)
    
    trading_workload = {
        'latency_sla_ms': 10,       # 10ms SLA
        'working_set_gb': 8,        # In-memory data
        'connections': 100,         # Market connections
        'request_rate': 50000       # High frequency
    }
    
    config = advisor.analyze(
        workload_data=trading_workload,
        target=SystemType.JVM
    )
    
    print(f"Configuration: {config.explanation}")
    print("\nJVM flags:")
    print(" ".join(config.commands))
    
    # Scenario 2: Batch processing
    print("\n\n2. Batch Processing Application")
    print("-"*40)
    
    batch_workload = {
        'batch_size': 10000,        # Large batches
        'working_set_gb': 50,       # Large heap needed
        'connections': 10,          # Few threads
        'latency_sla_ms': None      # Throughput focused
    }
    
    config = advisor.analyze(
        workload_data=batch_workload,
        target=SystemType.JVM
    )
    
    print(f"Configuration: {config.explanation}")
    print("\nJVM flags:")
    print(" ".join(config.commands))


def example_container_tuning():
    """Tune container resources"""
    print("\n\n" + "="*60)
    print("Container Resource Tuning Example")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    # Microservice workload
    print("\n1. Microservice API")
    print("-"*40)
    
    microservice_workload = {
        'working_set_gb': 2,        # Small footprint
        'connections': 100,         # API connections
        'qps': 1000,               # Request rate
        'avg_request_kb': 10        # Small payloads
    }
    
    config = advisor.analyze(
        workload_data=microservice_workload,
        target=SystemType.CONTAINER
    )
    
    print(f"Configuration: {config.explanation}")
    print("\nDocker command:")
    print(config.commands[0])
    
    # Database container
    print("\n\n2. Database Container")
    print("-"*40)
    
    db_container_workload = {
        'working_set_gb': 16,       # Database cache
        'total_data_gb': 100,       # Total data
        'connections': 200,         # DB connections
        'type': 'database'          # Hint for type
    }
    
    config = advisor.analyze(
        workload_data=db_container_workload,
        target=SystemType.CONTAINER
    )
    
    print(f"Configuration: {config.explanation}")
    print(f"\nSettings: {json.dumps(config.settings, indent=2)}")


def example_kernel_tuning():
    """Tune kernel parameters"""
    print("\n\n" + "="*60)
    print("Linux Kernel Tuning Example")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    # High-throughput server
    print("\n1. High-Throughput Web Server")
    print("-"*40)
    
    web_workload = {
        'request_rate': 50000,      # 50K req/s
        'connections': 10000,       # Many concurrent
        'working_set_gb': 32,       # Page cache
        'read_ratio': 0.95          # Mostly reads
    }
    
    config = advisor.analyze(
        workload_data=web_workload,
        target=SystemType.KERNEL
    )
    
    print(f"Configuration: {config.explanation}")
    print("\nSysctl commands:")
    for cmd in config.commands:
        print(f"  {cmd}")


def example_ab_testing():
    """Compare configurations with A/B testing"""
    print("\n\n" + "="*60)
    print("A/B Testing Example")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    # Test different database configurations
    print("\nComparing database configurations for mixed workload:")
    print("-"*50)
    
    # Configuration A: Optimized for reads
    config_a = advisor.analyze(
        workload_data={
            'read_ratio': 0.8,
            'working_set_gb': 100,
            'total_data_gb': 1000,
            'qps': 10000
        },
        target=SystemType.DATABASE
    )
    
    # Configuration B: Optimized for writes  
    config_b = advisor.analyze(
        workload_data={
            'read_ratio': 0.2,
            'working_set_gb': 100,
            'total_data_gb': 1000,
            'qps': 10000
        },
        target=SystemType.DATABASE
    )
    
    # Run A/B test
    results = advisor.compare_configs([config_a, config_b], test_duration=60)
    
    print("\nA/B Test Results:")
    for i, result in enumerate(results):
        config_name = f"Config {'A' if i == 0 else 'B'}"
        print(f"\n{config_name}:")
        print(f"  Throughput: {result.metrics['throughput']:.0f} QPS")
        print(f"  Latency: {result.metrics['latency']:.1f} ms")
        print(f"  CPU Usage: {result.metrics['cpu_usage']:.1%}")
        print(f"  Memory Usage: {result.metrics['memory_usage']:.1%}")
        if result.winner:
            print(f"  *** WINNER ***")


def example_adaptive_configuration():
    """Show how configurations adapt to changing workloads"""
    print("\n\n" + "="*60)
    print("Adaptive Configuration Example")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    print("\nMonitoring workload changes over time:")
    print("-"*50)
    
    # Simulate workload evolution
    workload_phases = [
        ("Morning (low traffic)", {
            'qps': 100,
            'connections': 50,
            'working_set_gb': 10
        }),
        ("Noon (peak traffic)", {
            'qps': 5000,
            'connections': 500,
            'working_set_gb': 50
        }),
        ("Evening (analytics)", {
            'qps': 50,
            'connections': 20,
            'working_set_gb': 200,
            'analytics': True
        })
    ]
    
    for phase_name, workload in workload_phases:
        print(f"\n{phase_name}:")
        
        config = advisor.analyze(
            workload_data=workload,
            target=SystemType.APPLICATION
        )
        
        settings = config.settings
        print(f"  Thread pool: {settings['thread_pool_size']} threads")
        print(f"  Connection pool: {settings['connection_pool_size']} connections")
        print(f"  Cache size: {settings['cache_size']:,} entries")
        if 'batch_size' in settings:
            print(f"  Batch size: {settings['batch_size']}")


def main():
    """Run all examples"""
    example_postgresql_tuning()
    example_jvm_tuning()
    example_container_tuning()
    example_kernel_tuning()
    example_ab_testing()
    example_adaptive_configuration()
    
    print("\n\n" + "="*60)
    print("Configuration Advisor Examples Complete!")
    print("="*60)
    print("\nKey Insights:")
    print("- √n sizing appears in buffer pools and caches")
    print("- Workload characteristics drive configuration")
    print("- A/B testing validates improvements")
    print("- Configurations should adapt to changing workloads")
    print("="*60)


if __name__ == "__main__":
    main()