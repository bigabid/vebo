"""
Unit tests for the MetaRuleDetector and related classes.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
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
    create_mixed_dataframe,
    create_edge_case_dataframe
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
    
    def test_column_attributes_to_dict(self):
        """Test converting ColumnAttributes to dictionary."""
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
        
        attributes_dict = attributes.to_dict()
        
        assert attributes_dict["name"] == "test_column"
        assert attributes_dict["type_category"] == "numeric"
        assert attributes_dict["diversity_level"] == "medium"
        assert attributes_dict["nullability_level"] == "low"
        assert attributes_dict["unique_count"] == 50
        assert attributes_dict["total_count"] == 100
        assert attributes_dict["null_count"] == 0
        assert attributes_dict["most_common_value"] == 42
        assert attributes_dict["most_common_frequency"] == 10


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
        
        # Test mixed numeric series
        mixed_series = pd.Series([1, 2.5, 3, 4.7, 5])
        assert detector.detect_column_type_category(mixed_series) == TypeCategory.NUMERIC
        
        # Test numeric with nulls
        numeric_with_nulls = pd.Series([1, 2, np.nan, 4, 5])
        assert detector.detect_column_type_category(numeric_with_nulls) == TypeCategory.NUMERIC
    
    def test_detect_column_type_textual(self):
        """Test detecting textual column type."""
        detector = MetaRuleDetector()
        
        # Test string series
        string_series = pd.Series(['a', 'b', 'c', 'd', 'e'])
        assert detector._detect_column_type(string_series) == TypeCategory.TEXTUAL
        
        # Test mixed string series
        mixed_string_series = pd.Series(['hello', 'world', 'test', 'data'])
        assert detector._detect_column_type(mixed_string_series) == TypeCategory.TEXTUAL
        
        # Test string with nulls
        string_with_nulls = pd.Series(['a', 'b', None, 'd', 'e'])
        assert detector._detect_column_type(string_with_nulls) == TypeCategory.TEXTUAL
    
    def test_detect_column_type_boolean(self):
        """Test detecting boolean column type."""
        detector = MetaRuleDetector()
        
        # Test boolean series
        bool_series = pd.Series([True, False, True, False, True])
        assert detector._detect_column_type(bool_series) == TypeCategory.BOOLEAN
        
        # Test boolean with nulls
        bool_with_nulls = pd.Series([True, False, None, False, True])
        assert detector._detect_column_type(bool_with_nulls) == TypeCategory.BOOLEAN
    
    def test_detect_column_type_temporal(self):
        """Test detecting temporal column type."""
        detector = MetaRuleDetector()
        
        # Test datetime series
        datetime_series = pd.Series(pd.date_range('2023-01-01', periods=5))
        assert detector._detect_column_type(datetime_series) == TypeCategory.TEMPORAL
        
        # Test datetime with nulls
        datetime_with_nulls = pd.Series(pd.date_range('2023-01-01', periods=5))
        datetime_with_nulls.iloc[2] = None
        assert detector._detect_column_type(datetime_with_nulls) == TypeCategory.TEMPORAL
    
    def test_detect_column_type_categorical(self):
        """Test detecting categorical column type."""
        detector = MetaRuleDetector()
        
        # Test categorical series
        cat_series = pd.Series(['A', 'B', 'A', 'B', 'A'])
        assert detector._detect_column_type(cat_series) == TypeCategory.CATEGORICAL
        
        # Test categorical with nulls
        cat_with_nulls = pd.Series(['A', 'B', None, 'B', 'A'])
        assert detector._detect_column_type(cat_with_nulls) == TypeCategory.CATEGORICAL
    
    def test_detect_column_type_unknown(self):
        """Test detecting unknown column type."""
        detector = MetaRuleDetector()
        
        # Test empty series
        empty_series = pd.Series([])
        assert detector._detect_column_type(empty_series) == TypeCategory.UNKNOWN
        
        # Test series with all nulls
        all_nulls_series = pd.Series([None, None, None, None, None])
        assert detector._detect_column_type(all_nulls_series) == TypeCategory.UNKNOWN
    
    def test_detect_diversity_level_constant(self):
        """Test detecting constant diversity level."""
        detector = MetaRuleDetector()
        
        # Test constant series
        constant_series = pd.Series([1, 1, 1, 1, 1])
        assert detector._detect_diversity_level(constant_series) == DiversityLevel.CONSTANT
        
        # Test constant with nulls
        constant_with_nulls = pd.Series([1, 1, None, 1, 1])
        assert detector._detect_diversity_level(constant_with_nulls) == DiversityLevel.CONSTANT
    
    def test_detect_diversity_level_binary(self):
        """Test detecting binary diversity level."""
        detector = MetaRuleDetector()
        
        # Test binary series
        binary_series = pd.Series([0, 1, 0, 1, 0])
        assert detector._detect_diversity_level(binary_series) == DiversityLevel.BINARY
        
        # Test binary with nulls
        binary_with_nulls = pd.Series([0, 1, None, 1, 0])
        assert detector._detect_diversity_level(binary_with_nulls) == DiversityLevel.BINARY
    
    def test_detect_diversity_level_low(self):
        """Test detecting low diversity level."""
        detector = MetaRuleDetector()
        
        # Test low diversity series
        low_diversity_series = pd.Series([1, 2, 1, 2, 1])
        assert detector._detect_diversity_level(low_diversity_series) == DiversityLevel.LOW
    
    def test_detect_diversity_level_medium(self):
        """Test detecting medium diversity level."""
        detector = MetaRuleDetector()
        
        # Test medium diversity series
        medium_diversity_series = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        assert detector._detect_diversity_level(medium_diversity_series) == DiversityLevel.MEDIUM
    
    def test_detect_diversity_level_high(self):
        """Test detecting high diversity level."""
        detector = MetaRuleDetector()
        
        # Test high diversity series
        high_diversity_series = pd.Series(range(50))
        assert detector._detect_diversity_level(high_diversity_series) == DiversityLevel.HIGH
    
    def test_detect_diversity_level_distinctive(self):
        """Test detecting distinctive diversity level."""
        detector = MetaRuleDetector()
        
        # Test distinctive series (all unique values)
        distinctive_series = pd.Series(range(100))
        assert detector._detect_diversity_level(distinctive_series) == DiversityLevel.DISTINCTIVE
    
    def test_detect_nullability_level_empty(self):
        """Test detecting empty nullability level."""
        detector = MetaRuleDetector()
        
        # Test empty series
        empty_series = pd.Series([])
        assert detector._detect_nullability_level(empty_series) == NullabilityLevel.EMPTY
        
        # Test series with all nulls
        all_nulls_series = pd.Series([None, None, None, None, None])
        assert detector._detect_nullability_level(all_nulls_series) == NullabilityLevel.EMPTY
    
    def test_detect_nullability_level_low(self):
        """Test detecting low nullability level."""
        detector = MetaRuleDetector()
        
        # Test series with few nulls
        low_nulls_series = pd.Series([1, 2, 3, 4, 5, None])
        assert detector._detect_nullability_level(low_nulls_series) == NullabilityLevel.LOW
    
    def test_detect_nullability_level_medium(self):
        """Test detecting medium nullability level."""
        detector = MetaRuleDetector()
        
        # Test series with moderate nulls
        medium_nulls_series = pd.Series([1, None, 3, None, 5, None, 7, None])
        assert detector._detect_nullability_level(medium_nulls_series) == NullabilityLevel.MEDIUM
    
    def test_detect_nullability_level_high(self):
        """Test detecting high nullability level."""
        detector = MetaRuleDetector()
        
        # Test series with many nulls
        high_nulls_series = pd.Series([1, None, None, None, 5, None, None, None])
        assert detector._detect_nullability_level(high_nulls_series) == NullabilityLevel.HIGH
    
    def test_detect_nullability_level_full(self):
        """Test detecting full nullability level."""
        detector = MetaRuleDetector()
        
        # Test series with mostly nulls
        full_nulls_series = pd.Series([None, None, None, None, None, 1])
        assert detector._detect_nullability_level(full_nulls_series) == NullabilityLevel.FULL
    
    def test_analyze_column(self):
        """Test analyzing a single column."""
        detector = MetaRuleDetector()
        
        # Test numeric column
        numeric_series = pd.Series([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10])
        attributes = detector._analyze_column('test_numeric', numeric_series)
        
        assert isinstance(attributes, ColumnAttributes)
        assert attributes.name == 'test_numeric'
        assert attributes.type_category == TypeCategory.NUMERIC
        assert attributes.unique_count == 9  # 9 non-null unique values
        assert attributes.total_count == 10
        assert attributes.null_count == 1
        
        # Test textual column
        textual_series = pd.Series(['a', 'b', 'c', 'd', 'e', None, 'g', 'h', 'i', 'j'])
        attributes = detector._analyze_column('test_textual', textual_series)
        
        assert isinstance(attributes, ColumnAttributes)
        assert attributes.name == 'test_textual'
        assert attributes.type_category == TypeCategory.TEXTUAL
        assert attributes.unique_count == 9  # 9 non-null unique values
        assert attributes.total_count == 10
        assert attributes.null_count == 1
    
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
        
        boolean_attrs = attributes['boolean_col']
        assert boolean_attrs.type_category == TypeCategory.BOOLEAN
        assert boolean_attrs.unique_count == 2
        assert boolean_attrs.total_count == 5
        assert boolean_attrs.null_count == 0
        
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
        
        # Test with threshold DataFrame (should not sample)
        threshold_df = pd.DataFrame({'col1': range(100000)})
        assert detector.should_enable_sampling(threshold_df, 100000) is False
    
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
    
    def test_create_sample_with_different_seeds(self):
        """Test that different seeds produce different samples."""
        detector1 = MetaRuleDetector(seed=42)
        detector2 = MetaRuleDetector(seed=123)
        
        # Create test DataFrame
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': [f'value_{i}' for i in range(1000)]
        })
        
        # Create samples with different seeds
        sample1 = detector1.create_sample(df, 100)
        sample2 = detector2.create_sample(df, 100)
        
        # Samples should be different (very high probability)
        assert not sample1.equals(sample2)
    
    def test_create_sample_empty_dataframe(self):
        """Test creating sample from empty DataFrame."""
        detector = MetaRuleDetector()
        
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        # Test sampling
        sample = detector.create_sample(empty_df, 100)
        
        assert len(sample) == 0
        assert list(sample.columns) == []
    
    def test_create_sample_smaller_than_requested(self):
        """Test creating sample when DataFrame is smaller than requested size."""
        detector = MetaRuleDetector()
        
        # Create small DataFrame
        small_df = pd.DataFrame({'col1': range(10)})
        
        # Test sampling with larger size
        sample = detector.create_sample(small_df, 100)
        
        assert len(sample) == 10
        assert list(sample.columns) == ['col1']
    
    def test_analyze_dataframe_with_edge_cases(self):
        """Test analyzing DataFrame with edge cases."""
        detector = MetaRuleDetector()
        
        # Create DataFrame with edge cases
        df = create_edge_case_dataframe()
        
        attributes = detector.analyze_dataframe(df)
        
        assert isinstance(attributes, dict)
        assert len(attributes) > 0
        
        # Check that all columns are analyzed
        for col in df.columns:
            assert col in attributes
            assert isinstance(attributes[col], ColumnAttributes)
    
    def test_analyze_dataframe_with_mixed_types(self):
        """Test analyzing DataFrame with mixed data types."""
        detector = MetaRuleDetector()
        
        # Create DataFrame with mixed types
        df = create_mixed_dataframe()
        
        attributes = detector.analyze_dataframe(df)
        
        assert isinstance(attributes, dict)
        assert len(attributes) == len(df.columns)
        
        # Check that different types are detected correctly
        type_categories = [attrs.type_category for attrs in attributes.values()]
        assert TypeCategory.NUMERIC in type_categories
        assert TypeCategory.TEXTUAL in type_categories
        assert TypeCategory.TEMPORAL in type_categories
        assert TypeCategory.BOOLEAN in type_categories
    
    def test_analyze_dataframe_with_nulls(self):
        """Test analyzing DataFrame with null values."""
        detector = MetaRuleDetector()
        
        # Create DataFrame with nulls
        df = pd.DataFrame({
            'no_nulls': [1, 2, 3, 4, 5],
            'some_nulls': [1, None, 3, None, 5],
            'many_nulls': [1, None, None, None, 5],
            'all_nulls': [None, None, None, None, None]
        })
        
        attributes = detector.analyze_dataframe(df)
        
        # Check null counts
        assert attributes['no_nulls'].null_count == 0
        assert attributes['some_nulls'].null_count == 2
        assert attributes['many_nulls'].null_count == 3
        assert attributes['all_nulls'].null_count == 5
        
        # Check nullability levels
        assert attributes['no_nulls'].nullability_level == NullabilityLevel.LOW
        assert attributes['some_nulls'].nullability_level == NullabilityLevel.MEDIUM
        assert attributes['many_nulls'].nullability_level == NullabilityLevel.HIGH
        assert attributes['all_nulls'].nullability_level == NullabilityLevel.EMPTY


if __name__ == "__main__":
    pytest.main([__file__])
