"""
Unit tests for the VeboProfiler class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.profiler import VeboProfiler, ProfilingConfig, ProfilingResult
from vebo_profiler.core.rule_engine import RuleResult, RuleStatus
from vebo_profiler.core.meta_rules import ColumnAttributes, TypeCategory, DiversityLevel, NullabilityLevel
from tests.fixtures.sample_data import (
    create_numeric_dataframe, 
    create_textual_dataframe, 
    create_mixed_dataframe,
    get_sample_metadata
)


class TestProfilingConfig:
    """Test cases for ProfilingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProfilingConfig()
        
        assert config.enable_cross_column is True
        assert config.deepness_level == "standard"
        assert config.max_workers is None
        assert config.timeout_seconds == 300
        assert config.sample_size == 10000
        assert config.sampling_threshold == 100000
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProfilingConfig(
            enable_cross_column=False,
            deepness_level="deep",
            max_workers=8,
            timeout_seconds=600,
            sample_size=5000,
            sampling_threshold=50000,
            random_seed=123
        )
        
        assert config.enable_cross_column is False
        assert config.deepness_level == "deep"
        assert config.max_workers == 8
        assert config.timeout_seconds == 600
        assert config.sample_size == 5000
        assert config.sampling_threshold == 50000
        assert config.random_seed == 123


class TestProfilingResult:
    """Test cases for ProfilingResult class."""
    
    def test_profiling_result_creation(self):
        """Test creating a ProfilingResult object."""
        metadata = get_sample_metadata()
        summary = {"overall_score": 85.5, "quality_grade": "B"}
        column_analysis = {"col1": {"data_type": "numeric"}}
        cross_column_analysis = {"checks": []}
        table_level_analysis = {"checks": []}
        errors = []
        
        result = ProfilingResult(
            metadata=metadata,
            summary=summary,
            column_analysis=column_analysis,
            cross_column_analysis=cross_column_analysis,
            table_level_analysis=table_level_analysis,
            errors=errors
        )
        
        assert result.metadata == metadata
        assert result.summary == summary
        assert result.column_analysis == column_analysis
        assert result.cross_column_analysis == cross_column_analysis
        assert result.table_level_analysis == table_level_analysis
        assert result.errors == errors


class TestVeboProfiler:
    """Test cases for VeboProfiler class."""
    
    def test_profiler_initialization_default(self):
        """Test profiler initialization with default config."""
        profiler = VeboProfiler()
        
        assert profiler.config is not None
        assert profiler.config.enable_cross_column is True
        assert profiler.meta_detector is not None
        assert profiler.rule_engine is not None
        assert profiler.check_executor is not None
    
    def test_profiler_initialization_custom(self):
        """Test profiler initialization with custom config."""
        config = ProfilingConfig(
            enable_cross_column=False,
            deepness_level="basic",
            max_workers=2
        )
        
        profiler = VeboProfiler(config)
        
        assert profiler.config == config
        assert profiler.config.enable_cross_column is False
        assert profiler.config.deepness_level == "basic"
        assert profiler.config.max_workers == 2
    
    @patch('vebo_profiler.core.profiler.MetaRuleDetector')
    @patch('vebo_profiler.core.profiler.RuleEngine')
    @patch('vebo_profiler.core.profiler.CheckExecutor')
    def test_profiler_initialization_dependencies(self, mock_check_executor, mock_rule_engine, mock_meta_detector):
        """Test that profiler initializes all dependencies correctly."""
        config = ProfilingConfig(random_seed=123)
        
        profiler = VeboProfiler(config)
        
        # Verify MetaRuleDetector was initialized with correct seed
        mock_meta_detector.assert_called_once_with(seed=123)
        
        # Verify RuleEngine was initialized
        mock_rule_engine.assert_called_once()
        
        # Verify CheckExecutor was initialized with correct config
        mock_check_executor.assert_called_once()
    
    def test_profile_dataframe_basic(self):
        """Test basic dataframe profiling."""
        df = create_numeric_dataframe()
        profiler = VeboProfiler()
        
        # Mock the internal methods to avoid complex dependencies
        with patch.object(profiler.meta_detector, 'should_enable_sampling', return_value=False), \
             patch.object(profiler.meta_detector, 'analyze_dataframe') as mock_analyze, \
             patch.object(profiler, '_execute_column_checks', return_value={}), \
             patch.object(profiler, '_execute_cross_column_checks', return_value=[]), \
             patch.object(profiler, '_execute_table_checks', return_value=[]):
            
            # Mock column attributes
            mock_analyze.return_value = {
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
            
            result = profiler.profile_dataframe(df, filename="test.csv")
            
            assert isinstance(result, ProfilingResult)
            assert result.metadata is not None
            assert result.summary is not None
            assert result.column_analysis is not None
            assert result.cross_column_analysis is not None
            assert result.table_level_analysis is not None
            assert result.errors is not None
    
    def test_profile_dataframe_with_sampling(self):
        """Test dataframe profiling with sampling enabled."""
        df = create_numeric_dataframe()
        config = ProfilingConfig(sampling_threshold=50, sample_size=20)
        profiler = VeboProfiler(config)
        
        with patch.object(profiler.meta_detector, 'should_enable_sampling', return_value=True), \
             patch.object(profiler.meta_detector, 'create_sample') as mock_sample, \
             patch.object(profiler.meta_detector, 'analyze_dataframe') as mock_analyze, \
             patch.object(profiler, '_execute_column_checks', return_value={}), \
             patch.object(profiler, '_execute_cross_column_checks', return_value=[]), \
             patch.object(profiler, '_execute_table_checks', return_value=[]):
            
            # Mock sampling
            sample_df = df.head(20)
            mock_sample.return_value = sample_df
            
            # Mock column attributes
            mock_analyze.return_value = {
                'age': ColumnAttributes(
                    name='age',
                    type_category=TypeCategory.NUMERIC,
                    diversity_level=DiversityLevel.MEDIUM,
                    nullability_level=NullabilityLevel.LOW,
                    unique_count=20,
                    total_count=20,
                    null_count=0
                )
            }
            
            result = profiler.profile_dataframe(df, filename="test.csv")
            
            # Verify sampling was called
            mock_sample.assert_called_once_with(df, 20)
            
            # Verify result contains sampling info
            assert result.metadata['dataset_info']['sampling_info']['was_sampled'] is True
            assert result.metadata['dataset_info']['sampling_info']['sample_size'] == 20
    
    def test_execute_column_checks(self):
        """Test column-level check execution."""
        df = create_numeric_dataframe()
        profiler = VeboProfiler()
        
        # Mock column attributes
        column_attributes = {
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
        
        with patch.object(profiler.rule_engine, 'get_relevant_rules') as mock_get_rules, \
             patch.object(profiler.check_executor, 'execute_column_checks_parallel') as mock_execute:
            
            # Mock rules
            mock_rule = Mock()
            mock_get_rules.return_value = [mock_rule]
            
            # Mock execution results
            mock_result = RuleResult(
                rule_id="test_rule",
                rule_name="Test Rule",
                status=RuleStatus.PASSED,
                score=100.0,
                message="Test passed",
                details={},
                execution_time_ms=10.0,
                timestamp=datetime.now().isoformat()
            )
            mock_execute.return_value = [mock_result]
            
            result = profiler._execute_column_checks(df, column_attributes)
            
            assert 'age' in result
            assert len(result['age']) == 1
            assert result['age'][0].rule_id == "test_rule"
    
    def test_execute_cross_column_checks_disabled(self):
        """Test cross-column checks when disabled."""
        config = ProfilingConfig(enable_cross_column=False)
        profiler = VeboProfiler(config)
        
        df = create_mixed_dataframe()
        column_attributes = {}
        
        result = profiler._execute_cross_column_checks(df, column_attributes)
        
        assert result == []
    
    def test_execute_cross_column_checks_enabled(self):
        """Test cross-column checks when enabled."""
        profiler = VeboProfiler()
        
        df = create_mixed_dataframe()
        column_attributes = {
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
        
        with patch.object(profiler.rule_engine, 'get_rules_by_category') as mock_get_rules, \
             patch.object(profiler, '_generate_column_pairs') as mock_pairs, \
             patch.object(profiler.check_executor, 'execute_cross_column_checks_parallel') as mock_execute:
            
            # Mock cross-column rules
            mock_rule = Mock()
            mock_get_rules.return_value = [mock_rule]
            
            # Mock column pairs
            mock_pairs.return_value = [('age', 'salary')]
            
            # Mock execution results
            mock_result = RuleResult(
                rule_id="correlation_rule",
                rule_name="Correlation Check",
                status=RuleStatus.PASSED,
                score=85.0,
                message="Strong correlation found",
                details={},
                execution_time_ms=15.0,
                timestamp=datetime.now().isoformat()
            )
            mock_execute.return_value = [mock_result]
            
            result = profiler._execute_cross_column_checks(df, column_attributes)
            
            assert len(result) == 1
            assert result[0].rule_id == "correlation_rule"
    
    def test_generate_column_pairs(self):
        """Test column pair generation for cross-column analysis."""
        profiler = VeboProfiler()
        
        df = create_mixed_dataframe()
        column_attributes = {
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
            'name': ColumnAttributes(
                name='name',
                type_category=TypeCategory.TEXTUAL,
                diversity_level=DiversityLevel.DISTINCTIVE,
                nullability_level=NullabilityLevel.LOW,
                unique_count=100,
                total_count=100,
                null_count=0
            )
        }
        
        pairs = profiler._generate_column_pairs(df, column_attributes)
        
        # Should generate all possible pairs
        expected_pairs = [('age', 'salary'), ('age', 'name'), ('salary', 'name')]
        assert len(pairs) == 3
        assert all(pair in expected_pairs for pair in pairs)
    
    def test_compile_results(self):
        """Test result compilation."""
        profiler = VeboProfiler()
        
        original_df = create_numeric_dataframe()
        analyzed_df = original_df.head(50)
        filename = "test.csv"
        was_sampled = True
        
        column_attributes = {
            'age': ColumnAttributes(
                name='age',
                type_category=TypeCategory.NUMERIC,
                diversity_level=DiversityLevel.MEDIUM,
                nullability_level=NullabilityLevel.LOW,
                unique_count=25,
                total_count=50,
                null_count=0
            )
        }
        
        column_results = {
            'age': [
                RuleResult(
                    rule_id="null_check",
                    rule_name="Null Check",
                    status=RuleStatus.PASSED,
                    score=100.0,
                    message="No nulls found",
                    details={},
                    execution_time_ms=5.0,
                    timestamp=datetime.now().isoformat()
                )
            ]
        }
        
        cross_column_results = []
        table_results = []
        
        start_time = datetime.now().timestamp()
        end_time = start_time + 10.0
        duration = 10.0
        
        result = profiler._compile_results(
            original_df, analyzed_df, filename, was_sampled,
            column_attributes, column_results, cross_column_results,
            table_results, start_time, end_time, duration
        )
        
        assert isinstance(result, ProfilingResult)
        assert result.metadata['dataset_info']['filename'] == "test.csv"
        assert result.metadata['dataset_info']['rows'] == 100
        assert result.metadata['dataset_info']['sampling_info']['was_sampled'] is True
        assert result.metadata['execution_info']['duration_seconds'] == 10.0
        assert 'age' in result.column_analysis
        assert result.column_analysis['age']['data_type'] == 'numeric'
        assert result.column_analysis['age']['null_count'] == 0
        assert result.column_analysis['age']['unique_count'] == 25
    
    def test_save_result(self):
        """Test saving profiling results to file."""
        profiler = VeboProfiler()
        
        result = ProfilingResult(
            metadata=get_sample_metadata(),
            summary={"overall_score": 85.5, "quality_grade": "B"},
            column_analysis={"col1": {"data_type": "numeric"}},
            cross_column_analysis={"checks": []},
            table_level_analysis={"checks": []},
            errors=[]
        )
        
        with patch('builtins.open', mock_open()) as mock_file:
            profiler.save_result(result, "test_results.json")
            
            mock_file.assert_called_once_with("test_results.json", 'w')
            mock_file().write.assert_called_once()
    
    def test_to_json(self):
        """Test converting ProfilingResult to JSON."""
        profiler = VeboProfiler()
        
        result = ProfilingResult(
            metadata=get_sample_metadata(),
            summary={"overall_score": 85.5, "quality_grade": "B"},
            column_analysis={"col1": {"data_type": "numeric"}},
            cross_column_analysis={"checks": []},
            table_level_analysis={"checks": []},
            errors=[]
        )
        
        json_str = profiler.to_json(result)
        
        assert isinstance(json_str, str)
        assert "overall_score" in json_str
        assert "85.5" in json_str
        assert "quality_grade" in json_str
        assert "B" in json_str


def mock_open():
    """Mock open function for testing file operations."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open()


if __name__ == "__main__":
    pytest.main([__file__])
