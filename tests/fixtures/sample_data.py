"""
Sample data fixtures for testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def create_numeric_dataframe() -> pd.DataFrame:
    """Create a DataFrame with various numeric columns for testing."""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'score': np.random.uniform(0, 100, n_rows),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_rows),
        'price': np.random.exponential(100, n_rows),
        'count': np.random.poisson(5, n_rows),
        'percentage': np.random.uniform(0, 1, n_rows),
        'negative_values': np.random.normal(0, 10, n_rows),
        'zeros': np.zeros(n_rows),
        'ones': np.ones(n_rows),
        'with_nulls': np.random.choice([1, 2, 3, np.nan], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'all_nulls': [np.nan] * n_rows,
        'constant': [42] * n_rows,
        'binary': np.random.choice([0, 1], n_rows),
        'outliers': np.concatenate([np.random.normal(0, 1, 96), [100, -100, 200, -200]])
    }
    
    return pd.DataFrame(data)


def create_textual_dataframe() -> pd.DataFrame:
    """Create a DataFrame with various textual columns for testing."""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'name': [f'Person_{i}' for i in range(n_rows)],
        'email': [f'person{i}@example.com' for i in range(n_rows)],
        'phone': [f'+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(n_rows)],
        'address': [f'{np.random.randint(1, 9999)} Main St' for _ in range(n_rows)],
        'description': [f'Description for person {i}' for i in range(n_rows)],
        'category': np.random.choice(['A', 'B', 'C'], n_rows),
        'status': np.random.choice(['Active', 'Inactive'], n_rows),
        'empty_strings': [''] * n_rows,
        'whitespace': ['   '] * n_rows,
        'with_nulls': np.random.choice(['text1', 'text2', 'text3', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'all_nulls': [None] * n_rows,
        'constant': ['Same Value'] * n_rows,
        'binary_text': np.random.choice(['Yes', 'No'], n_rows),
        'long_text': ['This is a very long text string that contains multiple words and should be handled properly by the text analysis rules'] * n_rows,
        'special_chars': ['!@#$%^&*()'] * n_rows
    }
    
    return pd.DataFrame(data)


def create_mixed_dataframe() -> pd.DataFrame:
    """Create a DataFrame with mixed data types for testing."""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        # Numeric columns
        'id': range(1, n_rows + 1),
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'score': np.random.uniform(0, 100, n_rows),
        
        # Textual columns
        'name': [f'Person_{i}' for i in range(n_rows)],
        'email': [f'person{i}@example.com' for i in range(n_rows)],
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], n_rows),
        
        # Boolean columns
        'is_active': np.random.choice([True, False], n_rows),
        'is_manager': np.random.choice([True, False], n_rows),
        
        # DateTime columns
        'created_date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'last_login': pd.date_range('2023-01-01', periods=n_rows, freq='h'),
        
        # Columns with nulls
        'bonus': np.random.choice([1000, 2000, 3000, np.nan], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'notes': np.random.choice(['Note 1', 'Note 2', 'Note 3', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        
        # Constant column
        'company': ['Vebo Corp'] * n_rows,
        
        # Binary column
        'is_remote': np.random.choice([0, 1], n_rows)
    }
    
    return pd.DataFrame(data)


def create_titanic_like_dataframe() -> pd.DataFrame:
    """Create a Titanic-like DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    
    # Passenger classes
    pclass = np.random.choice([1, 2, 3], n_rows, p=[0.2, 0.3, 0.5])
    
    # Gender
    sex = np.random.choice(['male', 'female'], n_rows, p=[0.6, 0.4])
    
    # Age with some missing values
    age = np.random.normal(30, 15, n_rows)
    age = np.where(age < 0, np.nan, age)
    age = np.where(np.random.random(n_rows) < 0.1, np.nan, age)  # 10% missing
    
    # Survival based on class and gender
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
    
    return pd.DataFrame(data)


def create_edge_case_dataframe() -> pd.DataFrame:
    """Create a DataFrame with edge cases for testing."""
    data = {
        'empty_column': [],
        'single_value': [42],
        'two_values': [1, 2],
        'all_zeros': [0, 0, 0, 0, 0],
        'all_ones': [1, 1, 1, 1, 1],
        'all_nulls': [None, None, None, None, None],
        'all_empty_strings': ['', '', '', '', ''],
        'mixed_nulls': [1, None, 2, None, 3],
        'mixed_types': [1, 'text', 2.5, True, None],
        'very_long_strings': ['a' * 1000, 'b' * 1000, 'c' * 1000],
        'special_characters': ['!@#$%', '^&*()', '{}[]'],
        'unicode': ['café', 'naïve', 'résumé'],
        'numbers_as_strings': ['1', '2', '3', '4', '5'],
        'dates_as_strings': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'booleans': [True, False, True, False, True]
    }
    
    # Create DataFrame with varying lengths
    max_len = max(len(v) for v in data.values() if v)
    for key, values in data.items():
        if len(values) < max_len:
            data[key] = values + [None] * (max_len - len(values))
    
    return pd.DataFrame(data)


def get_sample_metadata() -> Dict[str, Any]:
    """Get sample metadata for testing."""
    return {
        "dataset_info": {
            "filename": "test_dataset.csv",
            "rows": 100,
            "columns": 10,
            "file_size_bytes": 5000,
            "sampling_info": {
                "was_sampled": False,
                "sample_size": 100,
                "sample_method": "none",
                "seed": 42
            }
        },
        "execution_info": {
            "start_time": "2023-01-01T00:00:00",
            "end_time": "2023-01-01T00:01:00",
            "duration_seconds": 60.0,
            "rules_processed": 5,
            "checks_executed": 25,
            "errors_encountered": 0
        },
        "configuration": {
            "enabled_categories": ["basic_stats", "numeric_stats"],
            "deepness_level": "standard",
            "cross_column_checks": True
        }
    }
