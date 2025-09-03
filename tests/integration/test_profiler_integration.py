"""
Integration tests for the complete Vebo profiler workflow.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, Mock

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler import VeboProfiler
from vebo_profiler.core.profiler import ProfilingConfig
from vebo_profiler.core.rule_engine import Rule, RulePriority, RuleComplexity
from tests.fixtures.sample_data import (
    create_numeric_dataframe, 
    create_textual_dataframe, 
    create_mixed_dataframe,
    create_titanic_like_dataframe
)


class TestProfilerIntegration:
    """Integration tests for the complete profiler workflow."""
    
    def test_complete_profiling_workflow_numeric_data(self):
        """Test complete profiling workflow with numeric data."""
        # Create test data
        df = create_numeric_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=True,
            deepness_level="standard",
            max_workers=2,
            sample_size=50,
            sampling_threshold=100
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_numeric.csv")
        
        # Verify result structure
        assert result is not None
        assert result.metadata is not None
        assert result.summary is not None
        assert result.column_analysis is not None
        assert result.cross_column_analysis is not None
        assert result.table_level_analysis is not None
        assert result.errors is not None
        
        # Verify metadata
        assert result.metadata['dataset_info']['filename'] == "test_numeric.csv"
        assert result.metadata['dataset_info']['rows'] == len(df)
        assert result.metadata['dataset_info']['columns'] == len(df.columns)
        
        # Verify column analysis
        assert len(result.column_analysis) == len(df.columns)
        for col_name in df.columns:
            assert col_name in result.column_analysis
            col_analysis = result.column_analysis[col_name]
            assert 'data_type' in col_analysis
            assert 'null_count' in col_analysis
            assert 'unique_count' in col_analysis
            assert 'checks' in col_analysis
        
        # Verify summary
        assert 'overall_score' in result.summary
        assert 'quality_grade' in result.summary
        assert 'critical_issues' in result.summary
        assert 'warnings' in result.summary
        assert 'recommendations' in result.summary
        
        # Verify scores are reasonable
        assert 0 <= result.summary['overall_score'] <= 100
        assert result.summary['quality_grade'] in ['A', 'B', 'C', 'D', 'F']
    
    def test_complete_profiling_workflow_textual_data(self):
        """Test complete profiling workflow with textual data."""
        # Create test data
        df = create_textual_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=False,  # Disable cross-column for simplicity
            deepness_level="basic",
            max_workers=1
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_textual.csv")
        
        # Verify result structure
        assert result is not None
        assert result.metadata is not None
        assert result.summary is not None
        assert result.column_analysis is not None
        
        # Verify textual columns are detected
        for col_name, col_analysis in result.column_analysis.items():
            if col_name in ['name', 'email', 'phone', 'address', 'description']:
                assert col_analysis['data_type'] == 'textual'
    
    def test_complete_profiling_workflow_mixed_data(self):
        """Test complete profiling workflow with mixed data types."""
        # Create test data
        df = create_mixed_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=True,
            deepness_level="standard",
            max_workers=2
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_mixed.csv")
        
        # Verify result structure
        assert result is not None
        assert result.metadata is not None
        assert result.summary is not None
        assert result.column_analysis is not None
        assert result.cross_column_analysis is not None
        
        # Verify different data types are detected
        data_types = [col_analysis['data_type'] for col_analysis in result.column_analysis.values()]
        assert 'numeric' in data_types
        assert 'textual' in data_types
        assert 'temporal' in data_types
        assert 'boolean' in data_types
    
    def test_profiling_with_sampling(self):
        """Test profiling with data sampling enabled."""
        # Create large test data
        df = create_numeric_dataframe()
        # Make it larger to trigger sampling
        large_df = pd.concat([df] * 20, ignore_index=True)
        
        # Configure profiler with sampling
        config = ProfilingConfig(
            enable_cross_column=True,
            deepness_level="standard",
            sample_size=100,
            sampling_threshold=500  # Lower threshold to trigger sampling
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(large_df, filename="test_sampling.csv")
        
        # Verify sampling was used
        assert result.metadata['dataset_info']['sampling_info']['was_sampled'] is True
        assert result.metadata['dataset_info']['sampling_info']['sample_size'] == 100
        assert result.metadata['dataset_info']['rows'] == len(large_df)  # Original size
    
    def test_profiling_with_custom_rules(self):
        """Test profiling with custom rules added."""
        # Create test data
        df = create_numeric_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=False,
            deepness_level="basic"
        )
        
        profiler = VeboProfiler(config)
        
        # Add custom rule
        custom_rule = Rule(
            id="custom_range_check",
            name="Custom Range Check",
            description="Check if values are within expected range",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium", "high"],
            nullability_levels=["low", "medium"],
            requires_cross_column=False,
            priority=RulePriority.HIGH,
            complexity=RuleComplexity.LOW,
            code_template="""
def check_range(series):
    min_val = series.min()
    max_val = series.max()
    in_range = (min_val >= 0) and (max_val <= 1000)
    return {
        'min_value': min_val,
        'max_value': max_val,
        'in_range': in_range,
        'score': 100 if in_range else 50
    }
""",
            parameters={},
            enabled=True
        )
        
        profiler.rule_engine.add_rule(custom_rule)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_custom_rules.csv")
        
        # Verify custom rule was executed
        rule_executed = False
        for col_name, col_analysis in result.column_analysis.items():
            for check in col_analysis['checks']:
                if check['rule_id'] == 'custom_range_check':
                    rule_executed = True
                    break
        
        assert rule_executed, "Custom rule was not executed"
    
    def test_profiling_error_handling(self):
        """Test profiling with error handling."""
        # Create test data with problematic values
        df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'problematic_col': [1, 2, float('inf'), float('-inf'), 5],  # Infinity values
            'mixed_types': [1, 'text', 3, None, 5]  # Mixed types
        })
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=False,
            deepness_level="basic"
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_errors.csv")
        
        # Verify result is still generated despite errors
        assert result is not None
        assert result.metadata is not None
        assert result.summary is not None
        assert result.column_analysis is not None
        
        # Verify errors are captured
        assert result.errors is not None
        # Note: Errors might be in column analysis or in the errors list
    
    def test_profiling_result_serialization(self):
        """Test serializing profiling results to JSON."""
        # Create test data
        df = create_numeric_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=False,
            deepness_level="basic"
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_serialization.csv")
        
        # Test JSON serialization
        json_str = profiler.to_json(result)
        
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Verify JSON contains expected fields
        assert '"overall_score"' in json_str
        assert '"quality_grade"' in json_str
        assert '"column_analysis"' in json_str
        assert '"metadata"' in json_str
    
    def test_profiling_result_saving(self):
        """Test saving profiling results to file."""
        # Create test data
        df = create_numeric_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=False,
            deepness_level="basic"
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_saving.csv")
        
        # Test saving to file
        output_file = "test_output.json"
        
        with patch('builtins.open', mock_open()) as mock_file:
            profiler.save_result(result, output_file)
            
            mock_file.assert_called_once_with(output_file, 'w')
            mock_file().write.assert_called_once()
    
    def test_profiling_with_different_deepness_levels(self):
        """Test profiling with different deepness levels."""
        # Create test data
        df = create_mixed_dataframe()
        
        deepness_levels = ["basic", "standard", "deep"]
        
        for level in deepness_levels:
            # Configure profiler
            config = ProfilingConfig(
                enable_cross_column=True,
                deepness_level=level
            )
            
            profiler = VeboProfiler(config)
            
            # Profile the data
            result = profiler.profile_dataframe(df, filename=f"test_{level}.csv")
            
            # Verify result is generated
            assert result is not None
            assert result.metadata is not None
            assert result.summary is not None
            assert result.column_analysis is not None
            
            # Verify deepness level is recorded
            assert result.metadata['configuration']['deepness_level'] == level
    
    def test_profiling_with_titanic_like_data(self):
        """Test profiling with Titanic-like dataset."""
        # Create test data
        df = create_titanic_like_dataframe()
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=True,
            deepness_level="standard",
            max_workers=2
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data
        result = profiler.profile_dataframe(df, filename="test_titanic.csv")
        
        # Verify result structure
        assert result is not None
        assert result.metadata is not None
        assert result.summary is not None
        assert result.column_analysis is not None
        assert result.cross_column_analysis is not None
        
        # Verify Titanic-specific columns are analyzed
        expected_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
                          'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        
        for col in expected_columns:
            assert col in result.column_analysis
        
        # Verify data types are detected correctly
        assert result.column_analysis['PassengerId']['data_type'] == 'numeric'
        assert result.column_analysis['Survived']['data_type'] == 'numeric'
        assert result.column_analysis['Name']['data_type'] == 'textual'
        assert result.column_analysis['Sex']['data_type'] == 'textual'
        
        # Verify null handling for Age column
        age_analysis = result.column_analysis['Age']
        assert age_analysis['null_count'] > 0  # Age should have some nulls
        assert age_analysis['null_percentage'] > 0
    
    def test_profiling_performance(self):
        """Test profiling performance with larger dataset."""
        # Create larger test data
        df = create_numeric_dataframe()
        large_df = pd.concat([df] * 10, ignore_index=True)  # 1000 rows
        
        # Configure profiler
        config = ProfilingConfig(
            enable_cross_column=True,
            deepness_level="standard",
            max_workers=2,
            timeout_seconds=60
        )
        
        profiler = VeboProfiler(config)
        
        # Profile the data and measure time
        import time
        start_time = time.time()
        result = profiler.profile_dataframe(large_df, filename="test_performance.csv")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Verify result is generated
        assert result is not None
        assert result.metadata is not None
        
        # Verify execution time is reasonable (less than 60 seconds)
        assert execution_time < 60
        
        # Verify execution time is recorded
        assert result.metadata['execution_info']['duration_seconds'] > 0
        assert result.metadata['execution_info']['duration_seconds'] < 60


def mock_open():
    """Mock open function for testing file operations."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open()


if __name__ == "__main__":
    pytest.main([__file__])
