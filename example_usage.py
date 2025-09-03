#!/usr/bin/env python3
"""
Example usage of the Vebo Python code generation system for data profiling.

This script demonstrates how to use the profiler to analyze the Kaggle Titanic dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vebo_profiler import VeboProfiler
from vebo_profiler.core.profiler import ProfilingConfig



def download_titanic_dataset() -> pd.DataFrame:
    """
    Download and load the Kaggle Titanic dataset.
    
    Returns:
        Titanic DataFrame
    """
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download the dataset if not already present
    train_file = data_dir / "train.csv"
    if not train_file.exists():
        print("📥 Attempting to download Titanic dataset from Kaggle...")
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                'c/titanic', 
                path=str(data_dir), 
                unzip=True
            )
            print("✅ Dataset downloaded successfully!")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Please ensure you have:")
            print("1. Kaggle API credentials set up (kaggle.json in ~/.kaggle/)")
            print("2. kaggle package installed (pip install kaggle)")
            print("\nAlternatively, you can manually download the dataset from:")
            print("https://www.kaggle.com/c/titanic/data")
            print("and place train.csv in the 'data' directory.")
            print("\nFor now, using a sample dataset for demonstration...")
            # Create a sample Titanic-like dataset
            return create_sample_titanic_dataset()
    
    # Load the training data
    df = pd.read_csv(train_file)
    print(f"📊 Loaded Titanic dataset with {len(df)} rows and {len(df.columns)} columns")
    
    return df


def create_sample_titanic_dataset() -> pd.DataFrame:
    """
    Create a sample Titanic-like dataset for demonstration when Kaggle download fails.
    
    Returns:
        Sample Titanic DataFrame
    """
    np.random.seed(42)
    
    # Create sample data with Titanic-like structure
    n_rows = 1000
    
    # Passenger classes
    pclass = np.random.choice([1, 2, 3], n_rows, p=[0.2, 0.3, 0.5])
    
    # Gender
    sex = np.random.choice(['male', 'female'], n_rows, p=[0.6, 0.4])
    
    # Age with some missing values
    age = np.random.normal(30, 15, n_rows)
    age = np.where(age < 0, np.nan, age)
    age = np.where(np.random.random(n_rows) < 0.1, np.nan, age)  # 10% missing
    
    # Survival based on class and gender (higher class and female more likely to survive)
    survival_prob = np.where(sex == 'female', 0.7, 0.2)
    survival_prob = np.where(pclass == 1, survival_prob + 0.2, survival_prob)
    survival_prob = np.where(pclass == 3, survival_prob - 0.1, survival_prob)
    survived = np.random.binomial(1, np.clip(survival_prob, 0, 1), n_rows)
    
    # Fare based on class
    fare = np.where(pclass == 1, np.random.normal(100, 50, n_rows),
                   np.where(pclass == 2, np.random.normal(30, 15, n_rows),
                           np.random.normal(10, 5, n_rows)))
    fare = np.where(fare < 0, 0, fare)
    
    # Siblings/spouses and parents/children
    sibsp = np.random.poisson(0.5, n_rows)
    parch = np.random.poisson(0.4, n_rows)
    
    # Embarked ports
    embarked = np.random.choice(['S', 'C', 'Q'], n_rows, p=[0.7, 0.2, 0.1])
    
    # Names (simplified)
    names = [f"Passenger_{i}" for i in range(n_rows)]
    
    # Tickets
    tickets = [f"TICKET_{np.random.randint(100000, 999999)}" for _ in range(n_rows)]
    
    # Cabins (mostly missing)
    cabins = [f"C{np.random.randint(1, 100)}" if np.random.random() < 0.2 else None for _ in range(n_rows)]
    
    data = {
        'PassengerId': range(1, n_rows + 1),
        'Survived': survived,
        'Pclass': pclass,
        'Name': names,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Ticket': tickets,
        'Fare': fare,
        'Cabin': cabins,
        'Embarked': embarked
    }
    
    df = pd.DataFrame(data)
    print(f"📊 Created sample Titanic dataset with {len(df)} rows and {len(df.columns)} columns")
    
    return df


def main():
    """Main function to demonstrate the profiler."""
    print("🚀 Vebo Data Profiler - Python Code Generation System")
    print("=" * 60)
    
    # Download and load Titanic dataset
    print("📊 Loading Titanic dataset...")
    df = download_titanic_dataset()
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
        result = profiler.profile_dataframe(df, filename="titanic_train.csv")
        
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
        output_file = "titanic_profiling_results.json"
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
