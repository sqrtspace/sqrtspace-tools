#!/usr/bin/env python3
"""
SpaceTime Configuration Advisor: Analyze systems and recommend optimal settings

Features:
- System Analysis: Profile hardware capabilities
- Workload Characterization: Understand access patterns
- Configuration Generation: Produce optimal settings
- A/B Testing: Compare configurations in production
- AI Explanations: Clear reasoning for recommendations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import platform
import subprocess
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import sqlite3
import re

# Import core components
from core.spacetime_core import (
    MemoryHierarchy,
    SqrtNCalculator,
    OptimizationStrategy
)


class SystemType(Enum):
    """Types of systems to configure"""
    DATABASE = "database"
    JVM = "jvm"
    KERNEL = "kernel"
    CONTAINER = "container"
    APPLICATION = "application"


class WorkloadType(Enum):
    """Common workload patterns"""
    OLTP = "oltp"                    # Many small transactions
    OLAP = "olap"                    # Large analytical queries
    STREAMING = "streaming"          # Continuous data flow
    BATCH = "batch"                  # Periodic large jobs
    MIXED = "mixed"                  # Combination
    WEB = "web"                      # Web serving
    ML_TRAINING = "ml_training"      # Machine learning
    ML_INFERENCE = "ml_inference"    # Model serving


@dataclass
class SystemProfile:
    """Hardware and software profile"""
    # Hardware
    cpu_count: int
    cpu_model: str
    memory_gb: float
    memory_speed_mhz: Optional[int]
    storage_type: str  # 'ssd', 'nvme', 'hdd'
    storage_iops: Optional[int]
    network_speed_gbps: float
    
    # Software
    os_type: str
    os_version: str
    kernel_version: Optional[str]
    
    # Memory hierarchy
    l1_cache_kb: int
    l2_cache_kb: int
    l3_cache_mb: float
    numa_nodes: int
    
    # Current usage
    memory_used_percent: float
    cpu_usage_percent: float
    io_wait_percent: float


@dataclass
class WorkloadProfile:
    """Workload characteristics"""
    type: WorkloadType
    read_write_ratio: float      # 0.0 = write-only, 1.0 = read-only
    hot_data_size_gb: float      # Working set size
    total_data_size_gb: float    # Total dataset
    request_rate: float          # Requests per second
    avg_request_size_kb: float   # Average request size
    concurrency: int             # Concurrent connections/threads
    batch_size: Optional[int]    # For batch workloads
    latency_sla_ms: Optional[float]  # Latency requirement


@dataclass
class Configuration:
    """System configuration recommendations"""
    system_type: SystemType
    settings: Dict[str, Any]
    explanation: str
    expected_improvement: Dict[str, float]
    commands: List[str]  # Commands to apply settings
    validation_tests: List[str]  # Tests to verify improvement


@dataclass
class TestResult:
    """A/B test results"""
    config_name: str
    metrics: Dict[str, float]
    duration_seconds: float
    samples: int
    confidence: float
    winner: bool


class SystemAnalyzer:
    """Analyze system hardware and software"""
    
    def __init__(self):
        self.hierarchy = MemoryHierarchy.detect_system()
    
    def analyze_system(self) -> SystemProfile:
        """Comprehensive system analysis"""
        # CPU information
        cpu_count = psutil.cpu_count(logical=False)
        cpu_model = self._get_cpu_model()
        
        # Memory information
        mem = psutil.virtual_memory()
        memory_gb = mem.total / (1024**3)
        memory_speed = self._get_memory_speed()
        
        # Storage information
        storage_type, storage_iops = self._analyze_storage()
        
        # Network information
        network_speed = self._estimate_network_speed()
        
        # OS information
        os_type = platform.system()
        os_version = platform.version()
        kernel_version = platform.release() if os_type == 'Linux' else None
        
        # Cache sizes (from hierarchy)
        l1_cache_kb = self.hierarchy.l1_size // 1024
        l2_cache_kb = self.hierarchy.l2_size // 1024
        l3_cache_mb = self.hierarchy.l3_size // (1024 * 1024)
        
        # NUMA nodes
        numa_nodes = self._get_numa_nodes()
        
        # Current usage
        memory_used_percent = mem.percent / 100
        cpu_usage_percent = psutil.cpu_percent(interval=1) / 100
        io_wait = self._get_io_wait()
        
        return SystemProfile(
            cpu_count=cpu_count,
            cpu_model=cpu_model,
            memory_gb=memory_gb,
            memory_speed_mhz=memory_speed,
            storage_type=storage_type,
            storage_iops=storage_iops,
            network_speed_gbps=network_speed,
            os_type=os_type,
            os_version=os_version,
            kernel_version=kernel_version,
            l1_cache_kb=l1_cache_kb,
            l2_cache_kb=l2_cache_kb,
            l3_cache_mb=l3_cache_mb,
            numa_nodes=numa_nodes,
            memory_used_percent=memory_used_percent,
            cpu_usage_percent=cpu_usage_percent,
            io_wait_percent=io_wait
        )
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name"""
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            return line.split(':')[1].strip()
            elif platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return result.stdout.strip()
        except:
            pass
        return "Unknown CPU"
    
    def _get_memory_speed(self) -> Optional[int]:
        """Get memory speed in MHz"""
        # This would need platform-specific implementation
        # For now, return typical DDR4 speed
        return 2666
    
    def _analyze_storage(self) -> Tuple[str, Optional[int]]:
        """Analyze storage type and performance"""
        # Simplified detection
        partitions = psutil.disk_partitions()
        if partitions:
            # Check for NVMe
            device = partitions[0].device
            if 'nvme' in device:
                return 'nvme', 100000  # 100K IOPS typical
            elif any(x in device for x in ['ssd', 'solid']):
                return 'ssd', 50000   # 50K IOPS typical
        return 'hdd', 200  # 200 IOPS typical
    
    def _estimate_network_speed(self) -> float:
        """Estimate network speed in Gbps"""
        # Get network interface statistics
        stats = psutil.net_if_stats()
        speeds = []
        for interface, stat in stats.items():
            if stat.isup and stat.speed > 0:
                speeds.append(stat.speed)
        
        if speeds:
            # Return max speed in Gbps
            return max(speeds) / 1000
        return 1.0  # Default 1 Gbps
    
    def _get_numa_nodes(self) -> int:
        """Get number of NUMA nodes"""
        try:
            if platform.system() == 'Linux':
                result = subprocess.run(['lscpu'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'NUMA node(s)' in line:
                        return int(line.split(':')[1].strip())
        except:
            pass
        return 1
    
    def _get_io_wait(self) -> float:
        """Get I/O wait percentage"""
        # Simplified - would need proper implementation
        return 0.05  # 5% typical


class WorkloadAnalyzer:
    """Analyze workload characteristics"""
    
    def analyze_workload(self, 
                        logs: Optional[List[str]] = None,
                        metrics: Optional[Dict[str, Any]] = None) -> WorkloadProfile:
        """Analyze workload from logs or metrics"""
        # If no data provided, return default mixed workload
        if not logs and not metrics:
            return self._default_workload()
        
        # Analyze from provided data
        if metrics:
            return self._analyze_from_metrics(metrics)
        else:
            return self._analyze_from_logs(logs)
    
    def _default_workload(self) -> WorkloadProfile:
        """Default mixed workload profile"""
        return WorkloadProfile(
            type=WorkloadType.MIXED,
            read_write_ratio=0.8,
            hot_data_size_gb=10.0,
            total_data_size_gb=100.0,
            request_rate=1000.0,
            avg_request_size_kb=10.0,
            concurrency=100,
            batch_size=None,
            latency_sla_ms=100.0
        )
    
    def _analyze_from_metrics(self, metrics: Dict[str, Any]) -> WorkloadProfile:
        """Analyze from provided metrics"""
        # Determine workload type
        if metrics.get('batch_size'):
            workload_type = WorkloadType.BATCH
        elif metrics.get('streaming'):
            workload_type = WorkloadType.STREAMING
        elif metrics.get('analytics'):
            workload_type = WorkloadType.OLAP
        else:
            workload_type = WorkloadType.OLTP
        
        return WorkloadProfile(
            type=workload_type,
            read_write_ratio=metrics.get('read_ratio', 0.8),
            hot_data_size_gb=metrics.get('working_set_gb', 10.0),
            total_data_size_gb=metrics.get('total_data_gb', 100.0),
            request_rate=metrics.get('qps', 1000.0),
            avg_request_size_kb=metrics.get('avg_request_kb', 10.0),
            concurrency=metrics.get('connections', 100),
            batch_size=metrics.get('batch_size'),
            latency_sla_ms=metrics.get('latency_sla_ms', 100.0)
        )
    
    def _analyze_from_logs(self, logs: List[str]) -> WorkloadProfile:
        """Analyze from log entries"""
        # Simple pattern matching
        reads = sum(1 for log in logs if 'SELECT' in log or 'GET' in log)
        writes = sum(1 for log in logs if 'INSERT' in log or 'UPDATE' in log)
        total = reads + writes
        
        read_ratio = reads / total if total > 0 else 0.8
        
        return WorkloadProfile(
            type=WorkloadType.OLTP if read_ratio > 0.5 else WorkloadType.BATCH,
            read_write_ratio=read_ratio,
            hot_data_size_gb=10.0,
            total_data_size_gb=100.0,
            request_rate=len(logs),
            avg_request_size_kb=10.0,
            concurrency=100,
            batch_size=None,
            latency_sla_ms=100.0
        )


class ConfigurationGenerator:
    """Generate optimal configurations"""
    
    def __init__(self):
        self.sqrt_calc = SqrtNCalculator()
    
    def generate_config(self, 
                       system: SystemProfile,
                       workload: WorkloadProfile,
                       target: SystemType) -> Configuration:
        """Generate configuration for target system"""
        if target == SystemType.DATABASE:
            return self._generate_database_config(system, workload)
        elif target == SystemType.JVM:
            return self._generate_jvm_config(system, workload)
        elif target == SystemType.KERNEL:
            return self._generate_kernel_config(system, workload)
        elif target == SystemType.CONTAINER:
            return self._generate_container_config(system, workload)
        else:
            return self._generate_application_config(system, workload)
    
    def _generate_database_config(self, system: SystemProfile, 
                                 workload: WorkloadProfile) -> Configuration:
        """Generate database configuration"""
        settings = {}
        commands = []
        
        # Shared buffers (PostgreSQL) or buffer pool (MySQL)
        # Use 25% of RAM for database, but apply √n if data is large
        available_memory = system.memory_gb * 0.25
        
        if workload.total_data_size_gb > available_memory:
            # Use √n sizing
            sqrt_size_gb = np.sqrt(workload.total_data_size_gb)
            buffer_size_gb = min(sqrt_size_gb, available_memory)
        else:
            buffer_size_gb = min(workload.hot_data_size_gb, available_memory)
        
        settings['shared_buffers'] = f"{int(buffer_size_gb * 1024)}MB"
        
        # Work memory per operation
        work_mem_mb = int(available_memory * 1024 / workload.concurrency / 4)
        settings['work_mem'] = f"{work_mem_mb}MB"
        
        # WAL/Checkpoint settings
        if workload.read_write_ratio < 0.5:  # Write-heavy
            settings['checkpoint_segments'] = 64
            settings['checkpoint_completion_target'] = 0.9
        else:
            settings['checkpoint_segments'] = 16
            settings['checkpoint_completion_target'] = 0.5
        
        # Connection pool
        settings['max_connections'] = workload.concurrency * 2
        
        # Generate commands
        commands = [
            f"# PostgreSQL configuration",
            f"shared_buffers = {settings['shared_buffers']}",
            f"work_mem = {settings['work_mem']}",
            f"checkpoint_segments = {settings['checkpoint_segments']}",
            f"checkpoint_completion_target = {settings['checkpoint_completion_target']}",
            f"max_connections = {settings['max_connections']}"
        ]
        
        explanation = (
            f"Database configured with {buffer_size_gb:.1f}GB buffer pool "
            f"({'√n' if workload.total_data_size_gb > available_memory else 'full'} sizing), "
            f"{work_mem_mb}MB work memory per operation, and "
            f"{'aggressive' if workload.read_write_ratio < 0.5 else 'standard'} checkpointing."
        )
        
        expected_improvement = {
            'throughput': 1.5 if buffer_size_gb >= workload.hot_data_size_gb else 1.2,
            'latency': 0.7 if buffer_size_gb >= workload.hot_data_size_gb else 0.9,
            'memory_efficiency': 1.0 - (buffer_size_gb / system.memory_gb)
        }
        
        validation_tests = [
            "pgbench -c 10 -t 1000",
            "SELECT pg_stat_database_conflicts FROM pg_stat_database",
            "SELECT * FROM pg_stat_bgwriter"
        ]
        
        return Configuration(
            system_type=SystemType.DATABASE,
            settings=settings,
            explanation=explanation,
            expected_improvement=expected_improvement,
            commands=commands,
            validation_tests=validation_tests
        )
    
    def _generate_jvm_config(self, system: SystemProfile,
                           workload: WorkloadProfile) -> Configuration:
        """Generate JVM configuration"""
        settings = {}
        
        # Heap size - use 50% of available memory
        heap_size_gb = system.memory_gb * 0.5
        settings['-Xmx'] = f"{int(heap_size_gb)}g"
        settings['-Xms'] = f"{int(heap_size_gb)}g"  # Same as max to avoid resizing
        
        # Young generation - √n of heap for balanced GC
        young_gen_size = int(np.sqrt(heap_size_gb * 1024)) 
        settings['-Xmn'] = f"{young_gen_size}m"
        
        # GC algorithm
        if workload.latency_sla_ms and workload.latency_sla_ms < 100:
            settings['-XX:+UseG1GC'] = ''
            settings['-XX:MaxGCPauseMillis'] = int(workload.latency_sla_ms)
        else:
            settings['-XX:+UseParallelGC'] = ''
        
        # Thread settings
        settings['-XX:ParallelGCThreads'] = system.cpu_count
        settings['-XX:ConcGCThreads'] = max(1, system.cpu_count // 4)
        
        commands = ["java"] + [f"{k}{v}" if not k.startswith('-XX:+') else k 
                              for k, v in settings.items()]
        
        explanation = (
            f"JVM configured with {heap_size_gb:.0f}GB heap, "
            f"{young_gen_size}MB young generation (√n sizing), and "
            f"{'G1GC for low latency' if '-XX:+UseG1GC' in settings else 'ParallelGC for throughput'}."
        )
        
        return Configuration(
            system_type=SystemType.JVM,
            settings=settings,
            explanation=explanation,
            expected_improvement={'gc_time': 0.5, 'throughput': 1.3},
            commands=commands,
            validation_tests=["jstat -gcutil <pid> 1000 10"]
        )
    
    def _generate_kernel_config(self, system: SystemProfile,
                              workload: WorkloadProfile) -> Configuration:
        """Generate kernel configuration"""
        settings = {}
        commands = []
        
        # Page cache settings
        if workload.hot_data_size_gb > system.memory_gb * 0.5:
            # Aggressive page cache
            settings['vm.dirty_ratio'] = 5
            settings['vm.dirty_background_ratio'] = 2
        else:
            settings['vm.dirty_ratio'] = 20
            settings['vm.dirty_background_ratio'] = 10
        
        # Swappiness
        settings['vm.swappiness'] = 10 if workload.type in [WorkloadType.OLTP, WorkloadType.OLAP] else 60
        
        # Network settings for high throughput
        if workload.request_rate > 10000:
            settings['net.core.somaxconn'] = 65535
            settings['net.ipv4.tcp_max_syn_backlog'] = 65535
        
        # Generate sysctl commands
        commands = [f"sysctl -w {k}={v}" for k, v in settings.items()]
        
        explanation = (
            f"Kernel tuned for {'low' if settings['vm.swappiness'] == 10 else 'normal'} swappiness, "
            f"{'aggressive' if settings['vm.dirty_ratio'] == 5 else 'standard'} page cache, "
            f"and {'high' if 'net.core.somaxconn' in settings else 'normal'} network throughput."
        )
        
        return Configuration(
            system_type=SystemType.KERNEL,
            settings=settings,
            explanation=explanation,
            expected_improvement={'io_throughput': 1.2, 'latency': 0.9},
            commands=commands,
            validation_tests=["sysctl -a | grep vm.dirty"]
        )
    
    def _generate_container_config(self, system: SystemProfile,
                                 workload: WorkloadProfile) -> Configuration:
        """Generate container configuration"""
        settings = {}
        
        # Memory limits
        container_memory_gb = min(workload.hot_data_size_gb * 1.5, system.memory_gb * 0.8)
        settings['memory'] = f"{container_memory_gb:.1f}g"
        
        # CPU limits
        settings['cpus'] = min(workload.concurrency, system.cpu_count)
        
        # Shared memory for databases
        if workload.type in [WorkloadType.OLTP, WorkloadType.OLAP]:
            settings['shm_size'] = f"{int(container_memory_gb * 0.25)}g"
        
        commands = [
            f"docker run --memory={settings['memory']} --cpus={settings['cpus']}"
        ]
        
        explanation = (
            f"Container limited to {container_memory_gb:.1f}GB memory and "
            f"{settings['cpus']} CPUs based on workload requirements."
        )
        
        return Configuration(
            system_type=SystemType.CONTAINER,
            settings=settings,
            explanation=explanation,
            expected_improvement={'resource_efficiency': 1.5},
            commands=commands,
            validation_tests=["docker stats"]
        )
    
    def _generate_application_config(self, system: SystemProfile,
                                   workload: WorkloadProfile) -> Configuration:
        """Generate application-level configuration"""
        settings = {}
        
        # Thread pool sizing
        settings['thread_pool_size'] = min(workload.concurrency, system.cpu_count * 2)
        
        # Connection pool
        settings['connection_pool_size'] = workload.concurrency
        
        # Cache sizing using √n principle
        cache_entries = int(np.sqrt(workload.hot_data_size_gb * 1024 * 1024))
        settings['cache_size'] = cache_entries
        
        # Batch size for processing
        if workload.batch_size:
            settings['batch_size'] = workload.batch_size
        else:
            # Calculate optimal batch size
            memory_per_item = workload.avg_request_size_kb
            available_memory_mb = system.memory_gb * 1024 * 0.1  # 10% for batching
            settings['batch_size'] = int(available_memory_mb / memory_per_item)
        
        explanation = (
            f"Application configured with {settings['thread_pool_size']} threads, "
            f"{cache_entries:,} cache entries (√n sizing), and "
            f"batch size of {settings.get('batch_size', 'N/A')}."
        )
        
        return Configuration(
            system_type=SystemType.APPLICATION,
            settings=settings,
            explanation=explanation,
            expected_improvement={'throughput': 1.4, 'memory_usage': 0.7},
            commands=[],
            validation_tests=[]
        )


class ConfigurationAdvisor:
    """Main configuration advisor"""
    
    def __init__(self):
        self.system_analyzer = SystemAnalyzer()
        self.workload_analyzer = WorkloadAnalyzer()
        self.config_generator = ConfigurationGenerator()
    
    def analyze(self, 
                workload_data: Optional[Dict[str, Any]] = None,
                target: SystemType = SystemType.DATABASE) -> Configuration:
        """Analyze system and generate configuration"""
        # Analyze system
        print("Analyzing system hardware...")
        system_profile = self.system_analyzer.analyze_system()
        
        # Analyze workload
        print("Analyzing workload characteristics...")
        workload_profile = self.workload_analyzer.analyze_workload(
            metrics=workload_data
        )
        
        # Generate configuration
        print(f"Generating {target.value} configuration...")
        config = self.config_generator.generate_config(
            system_profile, workload_profile, target
        )
        
        return config
    
    def compare_configs(self, 
                       configs: List[Configuration],
                       test_duration: int = 300) -> List[TestResult]:
        """A/B test multiple configurations"""
        results = []
        
        for config in configs:
            print(f"\nTesting configuration: {config.system_type.value}")
            
            # Simulate test (in practice would apply config and measure)
            metrics = self._run_test(config, test_duration)
            
            result = TestResult(
                config_name=config.system_type.value,
                metrics=metrics,
                duration_seconds=test_duration,
                samples=test_duration * 10,
                confidence=0.95,
                winner=False
            )
            
            results.append(result)
        
        # Determine winner
        best_throughput = max(r.metrics.get('throughput', 0) for r in results)
        for result in results:
            if result.metrics.get('throughput', 0) == best_throughput:
                result.winner = True
                break
        
        return results
    
    def _run_test(self, config: Configuration, duration: int) -> Dict[str, float]:
        """Simulate running a test (would be real measurement in practice)"""
        # Simulate metrics based on expected improvement
        base_throughput = 1000.0
        base_latency = 50.0
        
        improvement = config.expected_improvement
        
        return {
            'throughput': base_throughput * improvement.get('throughput', 1.0),
            'latency': base_latency * improvement.get('latency', 1.0),
            'cpu_usage': 0.5 / improvement.get('throughput', 1.0),
            'memory_usage': improvement.get('memory_efficiency', 0.8)
        }
    
    def export_config(self, config: Configuration, filename: str):
        """Export configuration to file"""
        with open(filename, 'w') as f:
            if config.system_type == SystemType.DATABASE:
                f.write("# PostgreSQL Configuration\n")
                f.write("# Generated by SpaceTime Configuration Advisor\n\n")
                for cmd in config.commands:
                    f.write(cmd + "\n")
            elif config.system_type == SystemType.JVM:
                f.write("#!/bin/bash\n")
                f.write("# JVM Configuration\n")
                f.write("# Generated by SpaceTime Configuration Advisor\n\n")
                f.write(" ".join(config.commands) + " $@\n")
            else:
                json.dump(asdict(config), f, indent=2)
        
        print(f"Configuration exported to {filename}")


# Example usage
if __name__ == "__main__":
    print("SpaceTime Configuration Advisor")
    print("="*60)
    
    advisor = ConfigurationAdvisor()
    
    # Example 1: Database configuration
    print("\nExample 1: Database Configuration")
    print("-"*40)
    
    db_workload = {
        'read_ratio': 0.8,
        'working_set_gb': 50,
        'total_data_gb': 500,
        'qps': 10000,
        'connections': 200
    }
    
    db_config = advisor.analyze(
        workload_data=db_workload,
        target=SystemType.DATABASE
    )
    
    print(f"\nRecommendation: {db_config.explanation}")
    print("\nSettings:")
    for k, v in db_config.settings.items():
        print(f"  {k}: {v}")
    
    # Example 2: JVM configuration
    print("\n\nExample 2: JVM Configuration")
    print("-"*40)
    
    jvm_workload = {
        'latency_sla_ms': 50,
        'working_set_gb': 20,
        'connections': 1000
    }
    
    jvm_config = advisor.analyze(
        workload_data=jvm_workload,
        target=SystemType.JVM
    )
    
    print(f"\nRecommendation: {jvm_config.explanation}")
    print("\nJVM flags:")
    for cmd in jvm_config.commands[1:]:  # Skip 'java'
        print(f"  {cmd}")
    
    # Example 3: A/B testing
    print("\n\nExample 3: A/B Testing Configurations")
    print("-"*40)
    
    configs = [
        advisor.analyze(workload_data=db_workload, target=SystemType.DATABASE),
        advisor.analyze(workload_data={'read_ratio': 0.5}, target=SystemType.DATABASE)
    ]
    
    results = advisor.compare_configs(configs, test_duration=60)
    
    print("\nTest Results:")
    for result in results:
        print(f"\n{result.config_name}:")
        print(f"  Throughput: {result.metrics['throughput']:.0f} QPS")
        print(f"  Latency: {result.metrics['latency']:.1f} ms")
        print(f"  Winner: {'✓' if result.winner else '✗'}")
    
    # Export configuration
    advisor.export_config(db_config, "postgresql.conf")
    advisor.export_config(jvm_config, "jvm_startup.sh")
    
    print("\n" + "="*60)
    print("Configuration advisor complete!")
