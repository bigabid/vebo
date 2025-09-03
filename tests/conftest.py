"""
Pytest configuration and fixtures for Vebo profiler tests.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


@pytest.fixture
def sample_numeric_dataframe():
    """Fixture providing a sample numeric DataFrame."""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'score': np.random.uniform(0, 100, n_rows),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_rows),
        'with_nulls': np.random.choice([1, 2, 3, np.nan], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'constant': [42] * n_rows,
        'binary': np.random.choice([0, 1], n_rows)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_textual_dataframe():
    """Fixture providing a sample textual DataFrame."""
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'name': [f'Person_{i}' for i in range(n_rows)],
        'email': [f'person{i}@example.com' for i in range(n_rows)],
        'phone': [f'+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for _ in range(n_rows)],
        'category': np.random.choice(['A', 'B', 'C'], n_rows),
        'status': np.random.choice(['Active', 'Inactive'], n_rows),
        'with_nulls': np.random.choice(['text1', 'text2', 'text3', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'constant': ['Same Value'] * n_rows,
        'binary_text': np.random.choice(['Yes', 'No'], n_rows)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_mixed_dataframe():
    """Fixture providing a sample mixed-type DataFrame."""
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
        'last_login': pd.date_range('2023-01-01', periods=n_rows, freq='H'),
        
        # Columns with nulls
        'bonus': np.random.choice([1000, 2000, 3000, np.nan], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        'notes': np.random.choice(['Note 1', 'Note 2', 'Note 3', None], n_rows, p=[0.3, 0.3, 0.3, 0.1]),
        
        # Constant column
        'company': ['Vebo Corp'] * n_rows,
        
        # Binary column
        'is_remote': np.random.choice([0, 1], n_rows)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_titanic_dataframe():
    """Fixture providing a sample Titanic-like DataFrame."""
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


@pytest.fixture
def sample_edge_case_dataframe():
    """Fixture providing a DataFrame with edge cases."""
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


@pytest.fixture
def sample_profiling_config():
    """Fixture providing a sample profiling configuration."""
    from vebo_profiler.core.profiler import ProfilingConfig
    
    return ProfilingConfig(
        enable_cross_column=True,
        deepness_level="standard",
        max_workers=2,
        timeout_seconds=300,
        sample_size=1000,
        sampling_threshold=10000,
        random_seed=42
    )


@pytest.fixture
def sample_rule():
    """Fixture providing a sample rule."""
    from vebo_profiler.core.rule_engine import Rule, RulePriority, RuleComplexity
    
    return Rule(
        id="test_rule",
        name="Test Rule",
        description="A test rule for unit testing",
        category="data_quality",
        column_types=["numeric"],
        diversity_levels=["medium", "high"],
        nullability_levels=["low", "medium"],
        requires_cross_column=False,
        dependencies=[],
        priority=RulePriority.HIGH,
        complexity=RuleComplexity.LOW,
        code_template="def check_test(series):\n    return {'count': len(series), 'has_data': len(series) > 0}",
        parameters={},
        enabled=True
    )


@pytest.fixture
def sample_column_attributes():
    """Fixture providing sample column attributes."""
    from vebo_profiler.core.meta_rules import ColumnAttributes, TypeCategory, DiversityLevel, NullabilityLevel
    
    return ColumnAttributes(
        name="test_column",
        type_category=TypeCategory.NUMERIC,
        diversity_level=DiversityLevel.MEDIUM,
        nullability_level=NullabilityLevel.LOW,
        unique_count=50,
        total_count=100,
        null_count=0
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture providing a temporary directory for test outputs."""
    return tmp_path / "test_outputs"


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing the test data directory."""
    return Path(__file__).parent / "fixtures"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "performance: Performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker for tests that might take longer
        if "performance" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)
