"""
Unit tests for the CheckExecutor class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime
import time

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.check_executor import CheckExecutor, ExecutionConfig
from vebo_profiler.core.rule_engine import Rule, RuleResult, RuleStatus, RulePriority, RuleComplexity
from vebo_profiler.core.meta_rules import ColumnAttributes, TypeCategory, DiversityLevel, NullabilityLevel
from tests.fixtures.sample_data import create_numeric_dataframe, create_textual_dataframe, create_mixed_dataframe


class TestExecutionConfig:
    """Test cases for ExecutionConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()
        
        assert config.enable_cross_column is True
        assert config.max_workers is None
        assert config.timeout_seconds == 300
        assert config.enable_parallel is True
        assert config.sample_size == 10000
        assert config.sampling_threshold == 100000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExecutionConfig(
            enable_cross_column=False,
            max_workers=4,
            timeout_seconds=600,
            enable_parallel=False,
            sample_size=5000,
            sampling_threshold=50000
        )
        
        assert config.enable_cross_column is False
        assert config.max_workers == 4
        assert config.timeout_seconds == 600
        assert config.enable_parallel is False
        assert config.sample_size == 5000
        assert config.sampling_threshold == 50000


class TestCheckExecutor:
    """Test cases for CheckExecutor class."""
    
    def test_check_executor_initialization_default(self):
        """Test CheckExecutor initialization with default config."""
        executor = CheckExecutor()
        
        assert executor.config is not None
        assert executor.config.enable_cross_column is True
        assert executor.config.max_workers is not None
        assert executor.meta_detector is not None
    
    def test_check_executor_initialization_custom(self):
        """Test CheckExecutor initialization with custom config."""
        config = ExecutionConfig(
            enable_cross_column=False,
            max_workers=2,
            timeout_seconds=600
        )
        
        executor = CheckExecutor(config)
        
        assert executor.config == config
        assert executor.config.enable_cross_column is False
        assert executor.config.max_workers == 2
        assert executor.config.timeout_seconds == 600
    
    def test_execute_column_check_success(self):
        """Test successful column check execution."""
        executor = CheckExecutor()
        
        # Create test data
        series = pd.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])
        
        # Create test rule
        rule = Rule(
            id="null_check",
            name="Null Check",
            description="Check for null values",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low", "medium"],
            requires_cross_column=False,
            code_template="def check_nulls(series):\n    null_count = series.isnull().sum()\n    return {'null_count': null_count, 'has_nulls': null_count > 0}",
            parameters={}
        )
        
        # Create column attributes
        attributes = ColumnAttributes(
            name="test_column",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=9,
            total_count=10,
            null_count=1
        )
        
        result = executor.execute_column_check(rule, series, attributes)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "null_check"
        assert result.rule_name == "Null Check"
        assert result.status in [RuleStatus.PASSED, RuleStatus.FAILED, RuleStatus.WARNING]
        assert 0 <= result.score <= 100
        assert result.execution_time_ms > 0
        assert result.timestamp is not None
    
    def test_execute_column_check_with_error(self):
        """Test column check execution with error."""
        executor = CheckExecutor()
        
        # Create test data
        series = pd.Series([1, 2, 3, 4, 5])
        
        # Create test rule with invalid code
        rule = Rule(
            id="error_rule",
            name="Error Rule",
            description="Rule that will cause an error",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_error(series):\n    return undefined_variable",  # This will cause an error
            parameters={}
        )
        
        # Create column attributes
        attributes = ColumnAttributes(
            name="test_column",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=5,
            total_count=5,
            null_count=0
        )
        
        result = executor.execute_column_check(rule, series, attributes)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "error_rule"
        assert result.status == RuleStatus.ERROR
        assert result.score == 0
        assert "error" in result.message.lower()
    
    def test_execute_column_check_timeout(self):
        """Test column check execution with timeout."""
        config = ExecutionConfig(timeout_seconds=1)  # Very short timeout
        executor = CheckExecutor(config)
        
        # Create test data
        series = pd.Series([1, 2, 3, 4, 5])
        
        # Create test rule that will take too long
        rule = Rule(
            id="slow_rule",
            name="Slow Rule",
            description="Rule that takes too long",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_slow(series):\n    import time\n    time.sleep(2)\n    return {'result': 'done'}",
            parameters={}
        )
        
        # Create column attributes
        attributes = ColumnAttributes(
            name="test_column",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=5,
            total_count=5,
            null_count=0
        )
        
        result = executor.execute_column_check(rule, series, attributes)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "slow_rule"
        assert result.status == RuleStatus.ERROR
        assert "timeout" in result.message.lower()
    
    def test_execute_column_checks_parallel(self):
        """Test parallel execution of column checks."""
        executor = CheckExecutor()
        
        # Create test data
        series = pd.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])
        
        # Create test rules
        rule1 = Rule(
            id="null_check",
            name="Null Check",
            description="Check for null values",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low", "medium"],
            requires_cross_column=False,
            code_template="def check_nulls(series):\n    null_count = series.isnull().sum()\n    return {'null_count': null_count}",
            parameters={}
        )
        
        rule2 = Rule(
            id="range_check",
            name="Range Check",
            description="Check value range",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_range(series):\n    min_val = series.min()\n    max_val = series.max()\n    return {'min': min_val, 'max': max_val}",
            parameters={}
        )
        
        rules = [rule1, rule2]
        
        # Create column attributes
        attributes = ColumnAttributes(
            name="test_column",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=9,
            total_count=10,
            null_count=1
        )
        
        results = executor.execute_column_checks_parallel(rules, series, attributes)
        
        assert len(results) == 2
        assert all(isinstance(result, RuleResult) for result in results)
        assert all(result.rule_id in ["null_check", "range_check"] for result in results)
    
    def test_execute_cross_column_check(self):
        """Test cross-column check execution."""
        executor = CheckExecutor()
        
        # Create test DataFrame
        df = create_mixed_dataframe()
        
        # Create test rule
        rule = Rule(
            id="correlation_check",
            name="Correlation Check",
            description="Check correlation between columns",
            category="cross_column",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=True,
            code_template="def check_correlation(df, col1, col2):\n    corr = df[col1].corr(df[col2])\n    return {'correlation': corr, 'strong_correlation': abs(corr) > 0.7}",
            parameters={}
        )
        
        # Create column attributes
        attributes = {
            'age': ColumnAttributes(
                name='age',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.MEDIUM,
                nullability_level=NullabilityLevel.LOW,
                unique_count=50,
                total_count=100,
                null_count=0
            ),
            'salary': ColumnAttributes(
                name='salary',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.HIGH,
                nullability_level=NullabilityLevel.LOW,
                unique_count=80,
                total_count=100,
                null_count=0
            )
        }
        
        column_pair = ('age', 'salary')
        
        result = executor.execute_cross_column_check(rule, df, column_pair, attributes)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "correlation_check"
        assert result.status in [RuleStatus.PASSED, RuleStatus.FAILED, RuleStatus.WARNING]
        assert 0 <= result.score <= 100
        assert result.execution_time_ms > 0
    
    def test_execute_cross_column_checks_parallel(self):
        """Test parallel execution of cross-column checks."""
        executor = CheckExecutor()
        
        # Create test DataFrame
        df = create_mixed_dataframe()
        
        # Create test rule
        rule = Rule(
            id="correlation_check",
            name="Correlation Check",
            description="Check correlation between columns",
            category="cross_column",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=True,
            code_template="def check_correlation(df, col1, col2):\n    corr = df[col1].corr(df[col2])\n    return {'correlation': corr}",
            parameters={}
        )
        
        rules = [rule]
        
        # Create column attributes
        attributes = {
            'age': ColumnAttributes(
                name='age',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.MEDIUM,
                nullability_level=NullabilityLevel.LOW,
                unique_count=50,
                total_count=100,
                null_count=0
            ),
            'salary': ColumnAttributes(
                name='salary',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.HIGH,
                nullability_level=NullabilityLevel.LOW,
                unique_count=80,
                total_count=100,
                null_count=0
            ),
            'score': ColumnAttributes(
                name='score',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.HIGH,
                nullability_level=NullabilityLevel.LOW,
                unique_count=90,
                total_count=100,
                null_count=0
            )
        }
        
        column_pairs = [('age', 'salary'), ('age', 'score'), ('salary', 'score')]
        
        results = executor.execute_cross_column_checks_parallel(rules, df, column_pairs)
        
        assert len(results) == 3  # One result per column pair
        assert all(isinstance(result, RuleResult) for result in results)
        assert all(result.rule_id == "correlation_check" for result in results)
    
    def test_execute_table_check(self):
        """Test table-level check execution."""
        executor = CheckExecutor()
        
        # Create test DataFrame
        df = create_mixed_dataframe()
        
        # Create test rule
        rule = Rule(
            id="row_count_check",
            name="Row Count Check",
            description="Check number of rows",
            category="table_level",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_row_count(df):\n    row_count = len(df)\n    return {'row_count': row_count, 'has_data': row_count > 0}",
            parameters={}
        )
        
        # Create column attributes
        attributes = {
            'age': ColumnAttributes(
                name='age',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.MEDIUM,
                nullability_level=NullabilityLevel.LOW,
                unique_count=50,
                total_count=100,
                null_count=0
            )
        }
        
        result = executor.execute_table_check(rule, df, attributes)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "row_count_check"
        assert result.status in [RuleStatus.PASSED, RuleStatus.FAILED, RuleStatus.WARNING]
        assert 0 <= result.score <= 100
        assert result.execution_time_ms > 0
    
    def test_execute_table_checks_parallel(self):
        """Test parallel execution of table-level checks."""
        executor = CheckExecutor()
        
        # Create test DataFrame
        df = create_mixed_dataframe()
        
        # Create test rules
        rule1 = Rule(
            id="row_count_check",
            name="Row Count Check",
            description="Check number of rows",
            category="table_level",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_row_count(df):\n    return {'row_count': len(df)}",
            parameters={}
        )
        
        rule2 = Rule(
            id="column_count_check",
            name="Column Count Check",
            description="Check number of columns",
            category="table_level",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_column_count(df):\n    return {'column_count': len(df.columns)}",
            parameters={}
        )
        
        rules = [rule1, rule2]
        
        # Create column attributes
        attributes = {
            'age': ColumnAttributes(
                name='age',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.MEDIUM,
                nullability_level=NullabilityLevel.LOW,
                unique_count=50,
                total_count=100,
                null_count=0
            )
        }
        
        results = executor.execute_table_checks_parallel(rules, df, attributes)
        
        assert len(results) == 2
        assert all(isinstance(result, RuleResult) for result in results)
        assert all(result.rule_id in ["row_count_check", "column_count_check"] for result in results)
    
    def test_compile_check_function(self):
        """Test compilation of check function from code template."""
        executor = CheckExecutor()
        
        # Test valid code template
        code_template = """
def check_nulls(series):
    null_count = series.isnull().sum()
    return {'null_count': null_count, 'has_nulls': null_count > 0}
"""
        
        func = executor._compile_check_function(code_template)
        
        assert callable(func)
        
        # Test the function
        test_series = pd.Series([1, 2, np.nan, 4, 5])
        result = func(test_series)
        
        assert isinstance(result, dict)
        assert 'null_count' in result
        assert 'has_nulls' in result
        assert result['null_count'] == 1
        assert result['has_nulls'] is True
    
    def test_compile_check_function_invalid(self):
        """Test compilation of invalid check function."""
        executor = CheckExecutor()
        
        # Test invalid code template
        code_template = """
def check_invalid(series):
    return undefined_variable
"""
        
        func = executor._compile_check_function(code_template)
        
        # Should return None for invalid code
        assert func is None
    
    def test_evaluate_check_result(self):
        """Test evaluation of check results."""
        executor = CheckExecutor()
        
        # Test successful result
        result_data = {'null_count': 5, 'has_nulls': True}
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="Test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            parameters={}
        )
        
        result = executor._evaluate_check_result(result_data, rule, 10.0)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "test_rule"
        assert result.status in [RuleStatus.PASSED, RuleStatus.FAILED, RuleStatus.WARNING]
        assert 0 <= result.score <= 100
        assert result.execution_time_ms == 10.0
        assert result.details == result_data
    
    def test_evaluate_check_result_with_error(self):
        """Test evaluation of check results with error."""
        executor = CheckExecutor()
        
        # Test error result
        error_message = "Test error message"
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="Test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            parameters={}
        )
        
        result = executor._evaluate_check_result(None, rule, 5.0, error_message)
        
        assert isinstance(result, RuleResult)
        assert result.rule_id == "test_rule"
        assert result.status == RuleStatus.ERROR
        assert result.score == 0
        assert result.message == error_message
        assert result.execution_time_ms == 5.0
    
    def test_should_sample_data(self):
        """Test data sampling decision logic."""
        executor = CheckExecutor()
        
        # Test with small dataset (should not sample)
        small_df = pd.DataFrame({'col1': range(50)})
        assert executor._should_sample_data(small_df) is False
        
        # Test with large dataset (should sample)
        large_df = pd.DataFrame({'col1': range(150000)})
        assert executor._should_sample_data(large_df) is True
    
    def test_create_sample(self):
        """Test data sampling."""
        executor = CheckExecutor()
        
        # Create test DataFrame
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': [f'value_{i}' for i in range(1000)]
        })
        
        # Test sampling
        sample = executor._create_sample(df, 100)
        
        assert len(sample) == 100
        assert list(sample.columns) == list(df.columns)
        assert all(col in df.columns for col in sample.columns)
    
    def test_parallel_execution_disabled(self):
        """Test execution with parallel processing disabled."""
        config = ExecutionConfig(enable_parallel=False)
        executor = CheckExecutor(config)
        
        # Create test data
        series = pd.Series([1, 2, 3, 4, 5])
        
        # Create test rule
        rule = Rule(
            id="simple_check",
            name="Simple Check",
            description="Simple check",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            code_template="def check_simple(series):\n    return {'count': len(series)}",
            parameters={}
        )
        
        # Create column attributes
        attributes = ColumnAttributes(
            name="test_column",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=5,
            total_count=5,
            null_count=0
        )
        
        results = executor.execute_column_checks_parallel([rule], series, attributes)
        
        assert len(results) == 1
        assert isinstance(results[0], RuleResult)
        assert results[0].rule_id == "simple_check"


if __name__ == "__main__":
    pytest.main([__file__])
