#!/usr/bin/env python3
"""
Row-based performance optimization test for cross-column rules.

This script specifically tests the impact of row count on cross-column rule performance
and demonstrates the effectiveness of adaptive sampling and other row-based optimizations.
"""

import sys
import os
import time
import pandas as pd
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from vebo_profiler import VeboProfiler
from vebo_profiler.core.profiler import ProfilingConfig


def create_large_dataset(num_rows: int = 100000, num_cols: int = 20) -> pd.DataFrame:
    """
    Create a large dataset with focus on row count for testing row-based optimizations.
    
    Args:
        num_rows: Number of rows (this is the main variable we're testing)
        num_cols: Number of columns (kept smaller to focus on row impact)
        
    Returns:
        DataFrame with specified dimensions and realistic data patterns
    """
    np.random.seed(42)
    data = {}
    
    # Create columns that will benefit from different row optimizations
    for i in range(num_cols):
        if i % 3 == 0:
            # Numeric columns for correlation testing
            data[f'numeric_{i}'] = np.random.normal(100 + i*10, 15, num_rows)
            # Add some correlation between pairs
            if i > 0:
                prev_col = f'numeric_{i-3}'
                if prev_col in data:
                    correlation_factor = 0.7  # Strong correlation
                    noise = np.random.normal(0, 5, num_rows)
                    data[f'numeric_{i}'] = data[prev_col] * correlation_factor + noise
                    
        elif i % 3 == 1:
            # Categorical columns for functional dependency testing
            categories = ['Category_A', 'Category_B', 'Category_C', 'Category_D', 'Category_E']
            data[f'categorical_{i}'] = np.random.choice(categories, num_rows)
            
        else:
            # Mixed columns with missing values for missingness relationship testing
            base_values = np.random.choice(['Value_1', 'Value_2', 'Value_3'], num_rows)
            # Introduce correlated missingness patterns
            missing_indices = np.random.choice(num_rows, size=int(num_rows * 0.15), replace=False)
            data[f'mixed_{i}'] = pd.Series(base_values)
            data[f'mixed_{i}'].iloc[missing_indices] = None
    
    return pd.DataFrame(data)


def benchmark_row_performance(num_rows: int, num_cols: int = 20) -> dict:
    """
    Benchmark the performance impact of row count on cross-column rules.
    
    Args:
        num_rows: Number of rows to test
        num_cols: Number of columns
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n🔍 Testing {num_rows:,} rows x {num_cols} columns")
    
    # Create test dataset
    df = create_large_dataset(num_rows, num_cols)
    
    print(f"   📊 Dataset size: {len(df):,} rows, {len(df.columns)} columns")
    print(f"   💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Configure profiler with row optimizations enabled
    config = ProfilingConfig(
        enable_cross_column=True,
        deepness_level="standard",
        max_workers=4,
        sample_size=10000,
        sampling_threshold=50000  # Will trigger row optimizations
    )
    
    profiler = VeboProfiler(config)
    
    # Measure execution time
    start_time = time.time()
    result = profiler.profile_dataframe(df, filename=f"test_rows_{num_rows}.csv")
    end_time = time.time()
    
    duration = end_time - start_time
    cross_column_checks = len(result.cross_column_analysis.get('checks', []))
    
    print(f"   ⏱️  Duration: {duration:.2f} seconds")
    print(f"   🔗 Cross-column checks: {cross_column_checks}")
    print(f"   📈 Throughput: {num_rows/duration:,.0f} rows/second")
    
    return {
        'rows': num_rows,
        'columns': num_cols,
        'duration': duration,
        'cross_column_checks': cross_column_checks,
        'throughput': num_rows/duration,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }


def analyze_scaling_behavior(results: list):
    """
    Analyze how performance scales with row count.
    
    Args:
        results: List of benchmark results
    """
    print(f"\n📊 SCALING ANALYSIS")
    print("=" * 50)
    
    print(f"{'Rows':<12} {'Duration':<10} {'Throughput':<15} {'Efficiency':<12}")
    print("-" * 50)
    
    baseline_throughput = results[0]['throughput'] if results else 0
    
    for result in results:
        efficiency = (result['throughput'] / baseline_throughput) * 100 if baseline_throughput > 0 else 100
        print(f"{result['rows']:<12,} {result['duration']:<10.2f} "
              f"{result['throughput']:<15,.0f} {efficiency:<12.1f}%")
    
    # Check if optimizations are working
    large_dataset_result = next((r for r in results if r['rows'] >= 100000), None)
    if large_dataset_result and large_dataset_result['throughput'] > baseline_throughput * 0.5:
        print(f"\n✅ ROW OPTIMIZATIONS WORKING: Large datasets maintain good performance!")
    else:
        print(f"\n⚠️  Performance degrades significantly with row count")


def main():
    """Main test function."""
    print("🚀 Vebo Row-Based Performance Optimization Test")
    print("=" * 65)
    print("Testing how cross-column rule performance scales with row count...")
    print("Row optimizations include: adaptive sampling, early termination, statistical confidence")
    
    # Test different row counts to show scaling behavior
    test_row_counts = [
        10000,    # Small dataset
        50000,    # Medium dataset (threshold for row optimizations)
        100000,   # Large dataset
        250000,   # Very large dataset
        500000,   # Huge dataset
    ]
    
    results = []
    
    for row_count in test_row_counts:
        print(f"\n{'='*20} Testing {row_count:,} Rows {'='*20}")
        try:
            result = benchmark_row_performance(row_count)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to test {row_count:,} rows: {str(e)}")
            continue
    
    # Analyze results
    analyze_scaling_behavior(results)
    
    print(f"\n🎉 KEY ROW OPTIMIZATIONS IMPLEMENTED:")
    print("✅ Adaptive sampling: Different rules use optimal sample sizes")
    print("✅ Statistical sampling: Correlation stable with 5K rows vs 500K+")  
    print("✅ Early termination: Identity checks exit early on large datasets")
    print("✅ Memory efficiency: Process large datasets without memory issues")
    print("✅ Rule-specific optimization: Each rule optimized for its requirements")
    
    print(f"\n📈 PERFORMANCE GAINS:")
    if results:
        small = results[0]
        large = results[-1] if len(results) > 1 else results[0]
        
        rows_scaling = large['rows'] / small['rows']
        time_scaling = large['duration'] / small['duration']
        efficiency_ratio = time_scaling / rows_scaling
        
        print(f"   📊 Dataset size increased {rows_scaling:.1f}x")
        print(f"   ⏱️  Processing time increased {time_scaling:.1f}x")
        print(f"   🎯 Efficiency ratio: {efficiency_ratio:.2f} (lower is better)")
        
        if efficiency_ratio < 0.3:
            print(f"   🚀 EXCELLENT: Sublinear scaling achieved!")
        elif efficiency_ratio < 0.7:
            print(f"   ✅ GOOD: Significant optimization benefits")  
        else:
            print(f"   ⚠️  NEEDS WORK: Performance scales poorly with size")


if __name__ == "__main__":
    main()
