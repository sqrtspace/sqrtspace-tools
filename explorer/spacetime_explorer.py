#!/usr/bin/env python3
"""
Visual SpaceTime Explorer: Interactive visualization of space-time tradeoffs

Features:
- Interactive Plots: Pan, zoom, and explore tradeoff curves
- Live Updates: See impact of parameter changes in real-time
- Multiple Views: Memory hierarchy, checkpoint intervals, cache effects
- Export: Save visualizations and insights
- Educational: Understand theoretical bounds visually
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import time

# Import core components
from core.spacetime_core import (
    MemoryHierarchy,
    SqrtNCalculator,
    StrategyAnalyzer,
    OptimizationStrategy
)


class SpaceTimeVisualizer:
    """Main visualization engine"""
    
    def __init__(self):
        self.sqrt_calc = SqrtNCalculator()
        self.hierarchy = MemoryHierarchy.detect_system()
        self.strategy_analyzer = StrategyAnalyzer(self.hierarchy)
        
        # Plot settings
        self.fig = None
        self.axes = []
        self.animations = []
        
        # Data ranges
        self.n_min = 100
        self.n_max = 10**9
        self.n_points = 100
        
        # Current parameters
        self.current_n = 10**6
        self.current_strategy = 'sqrt_n'
        self.current_view = 'tradeoff'
        
    def create_main_window(self):
        """Create main visualization window"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('SpaceTime Explorer: Interactive Space-Time Tradeoff Visualization', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main tradeoff plot
        self.ax_tradeoff = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_tradeoff.set_title('Space-Time Tradeoff Curves')
        
        # Memory hierarchy view
        self.ax_hierarchy = self.fig.add_subplot(gs[0, 2])
        self.ax_hierarchy.set_title('Memory Hierarchy')
        
        # Checkpoint intervals
        self.ax_checkpoint = self.fig.add_subplot(gs[1, 2])
        self.ax_checkpoint.set_title('Checkpoint Intervals')
        
        # Cost analysis
        self.ax_cost = self.fig.add_subplot(gs[2, 0])
        self.ax_cost.set_title('Cost Analysis')
        
        # Performance metrics
        self.ax_metrics = self.fig.add_subplot(gs[2, 1])
        self.ax_metrics.set_title('Performance Metrics')
        
        # 3D visualization
        self.ax_3d = self.fig.add_subplot(gs[2, 2], projection='3d')
        self.ax_3d.set_title('3D Space-Time-Cost')
        
        # Add controls
        self._add_controls()
        
        # Initial plot
        self.update_all_plots()
        
    def _add_controls(self):
        """Add interactive controls"""
        # Sliders
        ax_n_slider = plt.axes([0.1, 0.02, 0.3, 0.02])
        self.n_slider = Slider(ax_n_slider, 'Data Size (log10)', 
                              np.log10(self.n_min), np.log10(self.n_max), 
                              valinit=np.log10(self.current_n), valstep=0.1)
        self.n_slider.on_changed(self._on_n_changed)
        
        # Strategy selector
        ax_strategy = plt.axes([0.5, 0.02, 0.15, 0.1])
        self.strategy_radio = RadioButtons(ax_strategy, 
                                         ['sqrt_n', 'linear', 'log_n', 'constant'],
                                         active=0)
        self.strategy_radio.on_clicked(self._on_strategy_changed)
        
        # View selector
        ax_view = plt.axes([0.7, 0.02, 0.15, 0.1])
        self.view_radio = RadioButtons(ax_view,
                                     ['tradeoff', 'animated', 'comparison'],
                                     active=0)
        self.view_radio.on_clicked(self._on_view_changed)
        
        # Export button
        ax_export = plt.axes([0.88, 0.02, 0.1, 0.04])
        self.export_btn = Button(ax_export, 'Export')
        self.export_btn.on_clicked(self._export_data)
        
    def update_all_plots(self):
        """Update all visualizations"""
        self.plot_tradeoff_curves()
        self.plot_memory_hierarchy()
        self.plot_checkpoint_intervals()
        self.plot_cost_analysis()
        self.plot_performance_metrics()
        self.plot_3d_visualization()
        
        plt.draw()
        
    def plot_tradeoff_curves(self):
        """Plot main space-time tradeoff curves"""
        self.ax_tradeoff.clear()
        
        # Generate data points
        n_values = np.logspace(np.log10(self.n_min), np.log10(self.n_max), self.n_points)
        
        # Theoretical bounds
        time_linear = n_values
        space_sqrt = np.sqrt(n_values * np.log(n_values))
        
        # Practical implementations
        strategies = {
            'O(n) space': (n_values, time_linear),
            'O(√n) space': (space_sqrt, time_linear * 1.5),
            'O(log n) space': (np.log(n_values), time_linear * n_values / 100),
            'O(1) space': (np.ones_like(n_values), time_linear ** 2)
        }
        
        # Plot curves
        for name, (space, time) in strategies.items():
            self.ax_tradeoff.loglog(space, time, label=name, linewidth=2)
        
        # Highlight current point
        current_space, current_time = self._get_current_point()
        self.ax_tradeoff.scatter(current_space, current_time, 
                               color='red', s=200, zorder=5, 
                               edgecolors='black', linewidth=2)
        
        # Theoretical bound (Williams)
        self.ax_tradeoff.fill_between(space_sqrt, time_linear * 0.9, time_linear * 50,
                                    alpha=0.2, color='gray', 
                                    label='Feasible region (Williams bound)')
        
        self.ax_tradeoff.set_xlabel('Space Usage')
        self.ax_tradeoff.set_ylabel('Time Complexity')
        self.ax_tradeoff.legend(loc='upper left')
        self.ax_tradeoff.grid(True, alpha=0.3)
        
        # Add annotations
        self.ax_tradeoff.annotate(f'Current: n={self.current_n:.0e}',
                                xy=(current_space, current_time),
                                xytext=(current_space*2, current_time*2),
                                arrowprops=dict(arrowstyle='->', color='red'))
        
    def plot_memory_hierarchy(self):
        """Visualize memory hierarchy and data placement"""
        self.ax_hierarchy.clear()
        
        # Memory levels
        levels = ['L1', 'L2', 'L3', 'RAM', 'SSD']
        sizes = [
            self.hierarchy.l1_size,
            self.hierarchy.l2_size,
            self.hierarchy.l3_size,
            self.hierarchy.ram_size,
            self.hierarchy.ssd_size
        ]
        latencies = [
            self.hierarchy.l1_latency_ns,
            self.hierarchy.l2_latency_ns,
            self.hierarchy.l3_latency_ns,
            self.hierarchy.ram_latency_ns,
            self.hierarchy.ssd_latency_ns
        ]
        
        # Calculate data distribution
        data_size = self.current_n * 8  # 8 bytes per element
        distribution = self._calculate_data_distribution(data_size, sizes)
        
        # Create stacked bar chart
        y_pos = np.arange(len(levels))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']
        
        bars = self.ax_hierarchy.barh(y_pos, distribution, color=colors)
        
        # Add size labels
        for i, (bar, size, dist) in enumerate(zip(bars, sizes, distribution)):
            if dist > 0:
                self.ax_hierarchy.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                                     f'{dist/size*100:.1f}%', 
                                     ha='center', va='center', fontsize=8)
        
        self.ax_hierarchy.set_yticks(y_pos)
        self.ax_hierarchy.set_yticklabels(levels)
        self.ax_hierarchy.set_xlabel('Data Distribution')
        self.ax_hierarchy.set_xlim(0, max(distribution) * 1.2)
        
        # Add latency annotations
        for i, (level, latency) in enumerate(zip(levels, latencies)):
            self.ax_hierarchy.text(max(distribution) * 1.1, i, f'{latency}ns',
                                 ha='left', va='center', fontsize=8)
        
    def plot_checkpoint_intervals(self):
        """Visualize checkpoint intervals for different strategies"""
        self.ax_checkpoint.clear()
        
        # Checkpoint strategies
        n = self.current_n
        strategies = {
            'No checkpoint': [n],
            '√n intervals': self._get_checkpoint_intervals(n, 'sqrt_n'),
            'Fixed 1000': self._get_checkpoint_intervals(n, 'fixed', 1000),
            'Exponential': self._get_checkpoint_intervals(n, 'exponential'),
        }
        
        # Plot timeline
        y_offset = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        for (name, intervals), color in zip(strategies.items(), colors):
            # Draw checkpoint blocks
            x_pos = 0
            for interval in intervals[:20]:  # Limit display
                rect = mpatches.Rectangle((x_pos, y_offset), interval, 0.8,
                                        facecolor=color, edgecolor='black', linewidth=0.5)
                self.ax_checkpoint.add_patch(rect)
                x_pos += interval
                if x_pos > n:
                    break
            
            # Label
            self.ax_checkpoint.text(-n*0.1, y_offset + 0.4, name,
                                  ha='right', va='center', fontsize=10)
            
            y_offset += 1
        
        self.ax_checkpoint.set_xlim(0, min(n, 10000))
        self.ax_checkpoint.set_ylim(-0.5, len(strategies) - 0.5)
        self.ax_checkpoint.set_xlabel('Progress')
        self.ax_checkpoint.set_yticks([])
        
        # Add checkpoint count
        for i, (name, intervals) in enumerate(strategies.items()):
            count = len(intervals)
            self.ax_checkpoint.text(min(n, 10000) * 1.05, i + 0.4, 
                                  f'{count} checkpoints',
                                  ha='left', va='center', fontsize=8)
        
    def plot_cost_analysis(self):
        """Analyze costs of different strategies"""
        self.ax_cost.clear()
        
        # Cost components
        strategies = ['O(n)', 'O(√n)', 'O(log n)', 'O(1)']
        memory_costs = [100, 10, 1, 0.1]
        time_costs = [1, 10, 100, 1000]
        total_costs = [m + t for m, t in zip(memory_costs, time_costs)]
        
        # Create grouped bar chart
        x = np.arange(len(strategies))
        width = 0.25
        
        bars1 = self.ax_cost.bar(x - width, memory_costs, width, label='Memory Cost')
        bars2 = self.ax_cost.bar(x, time_costs, width, label='Time Cost')
        bars3 = self.ax_cost.bar(x + width, total_costs, width, label='Total Cost')
        
        # Highlight current strategy
        current_idx = strategies.index(f'O({self.current_strategy.replace("_", " ")})')
        for bars in [bars1, bars2, bars3]:
            bars[current_idx].set_edgecolor('red')
            bars[current_idx].set_linewidth(3)
        
        self.ax_cost.set_xticks(x)
        self.ax_cost.set_xticklabels(strategies)
        self.ax_cost.set_ylabel('Relative Cost')
        self.ax_cost.legend()
        self.ax_cost.set_yscale('log')
        
    def plot_performance_metrics(self):
        """Show performance metrics for current configuration"""
        self.ax_metrics.clear()
        
        # Calculate metrics
        n = self.current_n
        metrics = self._calculate_performance_metrics(n, self.current_strategy)
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        self.ax_metrics.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
        self.ax_metrics.fill(angles, values, alpha=0.25, color='#4ECDC4')
        
        self.ax_metrics.set_xticks(angles[:-1])
        self.ax_metrics.set_xticklabels(categories, size=8)
        self.ax_metrics.set_ylim(0, 100)
        self.ax_metrics.grid(True)
        
        # Add value labels
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            self.ax_metrics.text(angle, value + 5, f'{value:.0f}', 
                               ha='center', va='center', size=8)
        
    def plot_3d_visualization(self):
        """3D visualization of space-time-cost tradeoffs"""
        self.ax_3d.clear()
        
        # Generate 3D surface
        n_range = np.logspace(2, 8, 20)
        strategies = ['sqrt_n', 'linear', 'log_n']
        
        for i, strategy in enumerate(strategies):
            space = []
            time = []
            cost = []
            
            for n in n_range:
                s, t, c = self._get_strategy_metrics(n, strategy)
                space.append(s)
                time.append(t)
                cost.append(c)
            
            self.ax_3d.plot(np.log10(space), np.log10(time), np.log10(cost),
                          label=strategy, linewidth=2)
        
        # Current point
        s, t, c = self._get_strategy_metrics(self.current_n, self.current_strategy)
        self.ax_3d.scatter([np.log10(s)], [np.log10(t)], [np.log10(c)],
                         color='red', s=100, edgecolors='black')
        
        self.ax_3d.set_xlabel('log₁₀(Space)')
        self.ax_3d.set_ylabel('log₁₀(Time)')
        self.ax_3d.set_zlabel('log₁₀(Cost)')
        self.ax_3d.legend()
        
    def create_animated_view(self):
        """Create animated visualization of algorithm progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Initialize plots
        n = 1000
        x = np.arange(n)
        y = np.random.rand(n)
        
        line1, = ax1.plot([], [], 'b-', label='Processing')
        checkpoint_lines = []
        
        ax1.set_xlim(0, n)
        ax1.set_ylim(0, 1)
        ax1.set_title('Algorithm Progress with Checkpoints')
        ax1.set_xlabel('Elements Processed')
        ax1.legend()
        
        # Memory usage over time
        line2, = ax2.plot([], [], 'r-', label='Memory Usage')
        ax2.set_xlim(0, n)
        ax2.set_ylim(0, n * 8 / 1024)  # KB
        ax2.set_title('Memory Usage Over Time')
        ax2.set_xlabel('Elements Processed')
        ax2.set_ylabel('Memory (KB)')
        ax2.legend()
        
        # Animation function
        checkpoint_interval = int(np.sqrt(n))
        memory_usage = []
        
        def animate(frame):
            # Update processing line
            line1.set_data(x[:frame], y[:frame])
            
            # Add checkpoint markers
            if frame % checkpoint_interval == 0 and frame > 0:
                checkpoint_line = ax1.axvline(x=frame, color='red', 
                                            linestyle='--', alpha=0.5)
                checkpoint_lines.append(checkpoint_line)
            
            # Update memory usage
            if self.current_strategy == 'sqrt_n':
                mem = min(frame, checkpoint_interval) * 8 / 1024
            else:
                mem = frame * 8 / 1024
            
            memory_usage.append(mem)
            line2.set_data(range(len(memory_usage)), memory_usage)
            
            return line1, line2
        
        anim = animation.FuncAnimation(fig, animate, frames=n, 
                                     interval=10, blit=True)
        
        plt.show()
        return anim
        
    def create_comparison_view(self):
        """Compare multiple strategies side by side"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        strategies = ['sqrt_n', 'linear', 'log_n', 'constant']
        n_range = np.logspace(2, 9, 100)
        
        for ax, strategy in zip(axes, strategies):
            # Calculate metrics
            space = []
            time = []
            
            for n in n_range:
                s, t, _ = self._get_strategy_metrics(n, strategy)
                space.append(s)
                time.append(t)
            
            # Plot
            ax.loglog(n_range, space, label='Space', linewidth=2)
            ax.loglog(n_range, time, label='Time', linewidth=2)
            ax.set_title(f'{strategy.replace("_", " ").title()} Strategy')
            ax.set_xlabel('Data Size (n)')
            ax.set_ylabel('Resource Usage')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add efficiency zone
            if strategy == 'sqrt_n':
                ax.axvspan(10**4, 10**7, alpha=0.2, color='green',
                          label='Optimal range')
        
        plt.tight_layout()
        plt.show()
        
    # Helper methods
    def _get_current_point(self) -> Tuple[float, float]:
        """Get current space-time point"""
        n = self.current_n
        
        if self.current_strategy == 'sqrt_n':
            space = np.sqrt(n * np.log(n))
            time = n * 1.5
        elif self.current_strategy == 'linear':
            space = n
            time = n
        elif self.current_strategy == 'log_n':
            space = np.log(n)
            time = n * n / 100
        else:  # constant
            space = 1
            time = n * n
            
        return space, time
        
    def _calculate_data_distribution(self, data_size: int, 
                                   memory_sizes: List[int]) -> List[float]:
        """Calculate how data is distributed across memory hierarchy"""
        distribution = []
        remaining = data_size
        
        for size in memory_sizes:
            if remaining <= 0:
                distribution.append(0)
            elif remaining <= size:
                distribution.append(remaining)
                remaining = 0
            else:
                distribution.append(size)
                remaining -= size
                
        return distribution
        
    def _get_checkpoint_intervals(self, n: int, strategy: str, 
                                param: Optional[int] = None) -> List[int]:
        """Get checkpoint intervals for different strategies"""
        if strategy == 'sqrt_n':
            interval = int(np.sqrt(n))
            return [interval] * (n // interval)
        elif strategy == 'fixed':
            interval = param or 1000
            return [interval] * (n // interval)
        elif strategy == 'exponential':
            intervals = []
            pos = 0
            exp = 1
            while pos < n:
                interval = min(2**exp, n - pos)
                intervals.append(interval)
                pos += interval
                exp += 1
            return intervals
        else:
            return [n]
            
    def _calculate_performance_metrics(self, n: int, 
                                     strategy: str) -> Dict[str, float]:
        """Calculate performance metrics"""
        # Base metrics
        if strategy == 'sqrt_n':
            memory_eff = 90
            speed = 70
            fault_tol = 85
            scalability = 95
            cost_eff = 80
        elif strategy == 'linear':
            memory_eff = 20
            speed = 100
            fault_tol = 50
            scalability = 40
            cost_eff = 60
        elif strategy == 'log_n':
            memory_eff = 95
            speed = 30
            fault_tol = 70
            scalability = 80
            cost_eff = 70
        else:  # constant
            memory_eff = 100
            speed = 10
            fault_tol = 60
            scalability = 90
            cost_eff = 50
            
        return {
            'Memory\nEfficiency': memory_eff,
            'Speed': speed,
            'Fault\nTolerance': fault_tol,
            'Scalability': scalability,
            'Cost\nEfficiency': cost_eff
        }
        
    def _get_strategy_metrics(self, n: int, 
                            strategy: str) -> Tuple[float, float, float]:
        """Get space, time, and cost for a strategy"""
        if strategy == 'sqrt_n':
            space = np.sqrt(n * np.log(n))
            time = n * 1.5
            cost = space * 0.1 + time * 0.01
        elif strategy == 'linear':
            space = n
            time = n
            cost = space * 0.1 + time * 0.01
        elif strategy == 'log_n':
            space = np.log(n)
            time = n * n / 100
            cost = space * 0.1 + time * 0.01
        else:  # constant
            space = 1
            time = n * n
            cost = space * 0.1 + time * 0.01
            
        return space, time, cost
        
    # Event handlers
    def _on_n_changed(self, val):
        """Handle data size slider change"""
        self.current_n = 10**val
        self.update_all_plots()
        
    def _on_strategy_changed(self, label):
        """Handle strategy selection change"""
        self.current_strategy = label
        self.update_all_plots()
        
    def _on_view_changed(self, label):
        """Handle view selection change"""
        self.current_view = label
        
        if label == 'animated':
            self.create_animated_view()
        elif label == 'comparison':
            self.create_comparison_view()
        else:
            self.update_all_plots()
            
    def _export_data(self, event):
        """Export visualization data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'spacetime_analysis_{timestamp}.json'
        
        data = {
            'timestamp': timestamp,
            'parameters': {
                'data_size': self.current_n,
                'strategy': self.current_strategy,
                'view': self.current_view
            },
            'metrics': self._calculate_performance_metrics(self.current_n, 
                                                          self.current_strategy),
            'space_time_point': self._get_current_point(),
            'system_info': {
                'l1_cache': self.hierarchy.l1_size,
                'l2_cache': self.hierarchy.l2_size,
                'l3_cache': self.hierarchy.l3_size,
                'ram_size': self.hierarchy.ram_size
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Exported analysis to {filename}")
        
        # Also save current figure
        self.fig.savefig(f'spacetime_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to spacetime_plot_{timestamp}.png")


def main():
    """Run the SpaceTime Explorer"""
    print("SpaceTime Explorer - Interactive Visualization")
    print("="*60)
    
    visualizer = SpaceTimeVisualizer()
    visualizer.create_main_window()
    
    print("\nControls:")
    print("- Slider: Adjust data size (n)")
    print("- Radio buttons: Select strategy and view")
    print("- Export: Save analysis and plots")
    print("- Mouse: Pan and zoom on plots")
    
    plt.show()


if __name__ == "__main__":
    main()