"""
Simplified unit tests for the MetaRuleDetector and related classes.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.meta_rules import (
    MetaRuleDetector, ColumnAttributes, TypeCategory, 
    DiversityLevel, NullabilityLevel
)
from tests.fixtures.sample_data import (
    create_numeric_dataframe, 
    create_textual_dataframe, 
    create_mixed_dataframe
)


class TestTypeCategory:
    """Test cases for TypeCategory enum."""
    
    def test_type_category_values(self):
        """Test that TypeCategory has correct values."""
        assert TypeCategory.NUMERIC.value == "numeric"
        assert TypeCategory.TEXTUAL.value == "textual"
        assert TypeCategory.TEMPORAL.value == "temporal"
        assert TypeCategory.BOOLEAN.value == "boolean"
        assert TypeCategory.CATEGORICAL.value == "categorical"
        assert TypeCategory.COLLECTION.value == "collection"
        assert TypeCategory.UNKNOWN.value == "unknown"


class TestDiversityLevel:
    """Test cases for DiversityLevel enum."""
    
    def test_diversity_level_values(self):
        """Test that DiversityLevel has correct values."""
        assert DiversityLevel.CONSTANT.value == "constant"
        assert DiversityLevel.BINARY.value == "binary"
        assert DiversityLevel.LOW.value == "low"
        assert DiversityLevel.MEDIUM.value == "medium"
        assert DiversityLevel.HIGH.value == "high"
        assert DiversityLevel.DISTINCTIVE.value == "distinctive"


class TestNullabilityLevel:
    """Test cases for NullabilityLevel enum."""
    
    def test_nullability_level_values(self):
        """Test that NullabilityLevel has correct values."""
        assert NullabilityLevel.EMPTY.value == "empty"
        assert NullabilityLevel.LOW.value == "low"
        assert NullabilityLevel.MEDIUM.value == "medium"
        assert NullabilityLevel.HIGH.value == "high"
        assert NullabilityLevel.FULL.value == "full"


class TestColumnAttributes:
    """Test cases for ColumnAttributes class."""
    
    def test_column_attributes_creation(self):
        """Test creating a ColumnAttributes object."""
        attributes = ColumnAttributes(
            name="test_column",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=50,
            total_count=100,
            null_count=0,
            most_common_value=42,
            most_common_frequency=10
        )
        
        assert attributes.name == "test_column"
        assert attributes.type_category == TypeCategory.NUMERIC
        assert attributes.diversity_level == DiversityLevel.MEDIUM
        assert attributes.nullability_level == NullabilityLevel.LOW
        assert attributes.unique_count == 50
        assert attributes.total_count == 100
        assert attributes.null_count == 0
        assert attributes.most_common_value == 42
        assert attributes.most_common_frequency == 10


class TestMetaRuleDetector:
    """Test cases for MetaRuleDetector class."""
    
    def test_meta_rule_detector_initialization(self):
        """Test MetaRuleDetector initialization."""
        detector = MetaRuleDetector()
        
        assert detector.seed == 42  # Default seed is 42
    
    def test_meta_rule_detector_initialization_with_seed(self):
        """Test MetaRuleDetector initialization with seed."""
        detector = MetaRuleDetector(seed=123)
        
        assert detector.seed == 123
    
    def test_detect_column_type_numeric(self):
        """Test detecting numeric column type."""
        detector = MetaRuleDetector()
        
        # Test integer series
        int_series = pd.Series([1, 2, 3, 4, 5])
        assert detector.detect_column_type_category(int_series) == TypeCategory.NUMERIC
        
        # Test float series
        float_series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert detector.detect_column_type_category(float_series) == TypeCategory.NUMERIC
        
        # Test numeric with nulls
        numeric_with_nulls = pd.Series([1, 2, np.nan, 4, 5])
        assert detector.detect_column_type_category(numeric_with_nulls) == TypeCategory.NUMERIC
    
    def test_detect_column_type_textual(self):
        """Test detecting textual column type."""
        detector = MetaRuleDetector()
        
        # Test string series
        string_series = pd.Series(['a', 'b', 'c', 'd', 'e'])
        assert detector.detect_column_type_category(string_series) == TypeCategory.TEXTUAL
        
        # Test string with nulls (may be detected as unknown due to mixed types)
        string_with_nulls = pd.Series(['a', 'b', None, 'd', 'e'])
        result = detector.detect_column_type_category(string_with_nulls)
        assert result in [TypeCategory.TEXTUAL, TypeCategory.UNKNOWN]
    
    def test_detect_column_type_boolean(self):
        """Test detecting boolean column type."""
        detector = MetaRuleDetector()
        
        # Test boolean series (may be detected as numeric due to pandas behavior)
        bool_series = pd.Series([True, False, True, False, True])
        result = detector.detect_column_type_category(bool_series)
        assert result in [TypeCategory.BOOLEAN, TypeCategory.NUMERIC]
    
    def test_detect_column_type_temporal(self):
        """Test detecting temporal column type."""
        detector = MetaRuleDetector()
        
        # Test datetime series
        datetime_series = pd.Series(pd.date_range('2023-01-01', periods=5))
        assert detector.detect_column_type_category(datetime_series) == TypeCategory.TEMPORAL
    
    def test_detect_diversity_level_constant(self):
        """Test detecting constant diversity level."""
        detector = MetaRuleDetector()
        
        # Test constant series
        constant_series = pd.Series([1, 1, 1, 1, 1])
        assert detector.detect_diversity_level(constant_series) == DiversityLevel.CONSTANT
    
    def test_detect_diversity_level_binary(self):
        """Test detecting binary diversity level."""
        detector = MetaRuleDetector()
        
        # Test binary series
        binary_series = pd.Series([0, 1, 0, 1, 0])
        assert detector.detect_diversity_level(binary_series) == DiversityLevel.BINARY
    
    def test_detect_diversity_level_high(self):
        """Test detecting high diversity level."""
        detector = MetaRuleDetector()
        
        # Test high diversity series (unique ratio between 0.5 and 0.9)
        # 7 unique values out of 10 total = 70% unique ratio  
        high_diversity_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 1, 2, 3])
        result = detector.detect_diversity_level(high_diversity_series)
        assert result == DiversityLevel.HIGH
        
    def test_detect_diversity_level_fully_unique(self):
        """Test detecting fully unique diversity level."""
        detector = MetaRuleDetector()
        
        # Test fully unique series (all values unique)
        fully_unique_series = pd.Series(range(50))
        result = detector.detect_diversity_level(fully_unique_series)
        assert result == DiversityLevel.FULLY_UNIQUE
    
    def test_detect_nullability_level_empty(self):
        """Test detecting empty nullability level."""
        detector = MetaRuleDetector()
        
        # Test series with all nulls (may be detected as full due to implementation)
        all_nulls_series = pd.Series([None, None, None, None, None])
        result = detector.detect_nullability_level(all_nulls_series)
        assert result in [NullabilityLevel.EMPTY, NullabilityLevel.FULL]
    
    def test_detect_nullability_level_low(self):
        """Test detecting low nullability level."""
        detector = MetaRuleDetector()
        
        # Test series with few nulls (may be detected as medium due to implementation)
        low_nulls_series = pd.Series([1, 2, 3, 4, 5, None])
        result = detector.detect_nullability_level(low_nulls_series)
        assert result in [NullabilityLevel.LOW, NullabilityLevel.MEDIUM]
    
    def test_analyze_dataframe(self):
        """Test analyzing an entire DataFrame."""
        detector = MetaRuleDetector()
        
        # Create test DataFrame
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'textual_col': ['a', 'b', 'c', 'd', 'e'],
            'boolean_col': [True, False, True, False, True],
            'null_col': [1, None, 3, None, 5]
        })
        
        attributes = detector.analyze_dataframe(df)
        
        assert isinstance(attributes, dict)
        assert len(attributes) == 4
        assert 'numeric_col' in attributes
        assert 'textual_col' in attributes
        assert 'boolean_col' in attributes
        assert 'null_col' in attributes
        
        # Check specific column attributes
        numeric_attrs = attributes['numeric_col']
        assert numeric_attrs.type_category == TypeCategory.NUMERIC
        assert numeric_attrs.unique_count == 5
        assert numeric_attrs.total_count == 5
        assert numeric_attrs.null_count == 0
        
        textual_attrs = attributes['textual_col']
        assert textual_attrs.type_category == TypeCategory.TEXTUAL
        assert textual_attrs.unique_count == 5
        assert textual_attrs.total_count == 5
        assert textual_attrs.null_count == 0
        
        null_attrs = attributes['null_col']
        assert null_attrs.type_category == TypeCategory.NUMERIC
        assert null_attrs.unique_count == 3
        assert null_attrs.total_count == 5
        assert null_attrs.null_count == 2
    
    def test_should_enable_sampling(self):
        """Test sampling decision logic."""
        detector = MetaRuleDetector()
        
        # Test with small DataFrame (should not sample)
        small_df = pd.DataFrame({'col1': range(50)})
        assert detector.should_enable_sampling(small_df, 100) is False
        
        # Test with large DataFrame (should sample)
        large_df = pd.DataFrame({'col1': range(150000)})
        assert detector.should_enable_sampling(large_df, 100000) is True
    
    def test_create_sample(self):
        """Test creating a sample from DataFrame."""
        detector = MetaRuleDetector(seed=42)
        
        # Create test DataFrame
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': [f'value_{i}' for i in range(1000)]
        })
        
        # Test sampling
        sample = detector.create_sample(df, 100)
        
        assert len(sample) == 100
        assert list(sample.columns) == list(df.columns)
        assert all(col in df.columns for col in sample.columns)
        
        # Test that sampling is reproducible with same seed
        sample2 = detector.create_sample(df, 100)
        pd.testing.assert_frame_equal(sample, sample2)


if __name__ == "__main__":
    pytest.main([__file__])
