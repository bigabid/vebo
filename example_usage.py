#!/usr/bin/env python3
"""
Example usage of the Vebo Python code generation system for data profiling.

This script demonstrates how to use the profiler to analyze a sample dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vebo_profiler import VeboProfiler, ProfilingConfig


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample dataset for demonstration.
    
    Returns:
        Sample DataFrame with various data types and patterns
    """
    np.random.seed(42)
    
    # Create sample data
    n_rows = 1000
    
    data = {
        # Numeric columns
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'score': np.random.uniform(0, 100, n_rows),
        
        # Textual columns
        'name': [f'Person_{i}' for i in range(n_rows)],
        'email': [f'person{i}@example.com' for i in range(n_rows)],
        'phone': [f'+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(n_rows)],
        
        # Categorical columns
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_rows),
        'status': np.random.choice(['Active', 'Inactive'], n_rows),
        
        # Boolean columns
        'is_manager': np.random.choice([True, False], n_rows),
        
        # DateTime columns
        'hire_date': [datetime.now() - timedelta(days=np.random.randint(0, 3650)) for _ in range(n_rows)],
        
        # Columns with nulls
        'bonus': np.random.choice([1000, 2000, 3000, np.nan], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        
        # Constant column
        'company': ['Vebo Corp'] * n_rows,
        
        # Binary column
        'is_remote': np.random.choice([0, 1], n_rows)
    }
    
    return pd.DataFrame(data)


def main():
    """Main function to demonstrate the profiler."""
    print("🚀 Vebo Data Profiler - Python Code Generation System")
    print("=" * 60)
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    df = create_sample_dataset()
    print(f"   Created dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Configure profiler
    print("⚙️  Configuring profiler...")
    config = ProfilingConfig(
        enable_cross_column=True,
        deepness_level="standard",
        max_workers=4,
        sample_size=500,  # Use smaller sample for demo
        sampling_threshold=100,  # Lower threshold for demo
        random_seed=42
    )
    
    profiler = VeboProfiler(config)
    print(f"   Deepness level: {config.deepness_level}")
    print(f"   Cross-column checks: {config.enable_cross_column}")
    print(f"   Max workers: {config.max_workers}")
    print()
    
    # Profile the dataset
    print("🔍 Profiling dataset...")
    start_time = datetime.now()
    
    try:
        result = profiler.profile_dataframe(df, filename="sample_dataset.csv")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"   ✅ Profiling completed in {duration:.2f} seconds")
        print()
        
        # Display summary
        print("📈 PROFILING SUMMARY")
        print("-" * 30)
        print(f"Overall Score: {result.summary['overall_score']:.1f}/100")
        print(f"Quality Grade: {result.summary['quality_grade']}")
        print(f"Critical Issues: {result.summary['critical_issues']}")
        print(f"Warnings: {result.summary['warnings']}")
        print(f"Recommendations: {result.summary['recommendations']}")
        print()
        
        # Display column analysis
        print("📋 COLUMN ANALYSIS")
        print("-" * 30)
        for col_name, analysis in result.column_analysis.items():
            print(f"Column: {col_name}")
            print(f"  Type: {analysis['data_type']}")
            print(f"  Nulls: {analysis['null_count']} ({analysis['null_percentage']:.1%})")
            print(f"  Unique: {analysis['unique_count']} ({analysis['unique_percentage']:.1%})")
            print(f"  Checks: {len(analysis['checks'])} executed")
            print()
        
        # Display cross-column analysis
        if result.cross_column_analysis['checks']:
            print("🔗 CROSS-COLUMN ANALYSIS")
            print("-" * 30)
            print(f"Cross-column checks: {len(result.cross_column_analysis['checks'])} executed")
            print()
        
        # Display errors
        if result.errors:
            print("❌ ERRORS")
            print("-" * 30)
            for error in result.errors:
                print(f"  {error['check_id']}: {error['message']}")
            print()
        
        # Save results
        output_file = "profiling_results.json"
        profiler.save_result(result, output_file)
        print(f"💾 Results saved to: {output_file}")
        
        # Display some example check results
        print("\n🔍 EXAMPLE CHECK RESULTS")
        print("-" * 30)
        for col_name, analysis in list(result.column_analysis.items())[:3]:  # Show first 3 columns
            print(f"Column: {col_name}")
            for check in analysis['checks'][:2]:  # Show first 2 checks
                print(f"  {check['name']}: {check['status']} - {check['message']}")
            print()
        
    except Exception as e:
        print(f"❌ Error during profiling: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("🎉 Demo completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
