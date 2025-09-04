"""
Unit tests for frontend filtering logic (simulated in Python).
This tests the core filtering algorithms used in the React component.
"""

import pytest
from typing import Dict, Set, List, Any, Optional


# Mock ColumnInsight structure
class ColumnInsight:
    def __init__(self, name: str, type: str, value_type: Optional[str] = None):
        self.name = name
        self.type = type
        self.value_type = value_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "valueType": self.value_type
        }


def filter_columns_by_type(
    columns: List[ColumnInsight], 
    selected_value_types: Set[str], 
    selected_data_types: Set[str]
) -> List[ColumnInsight]:
    """
    Simulate the filtering logic used in the React component.
    This matches the logic in InsightsPanel.tsx.
    """
    filtered_columns = []
    
    for column in columns:
        # Apply value type filter (if any selected)
        value_type_match = (
            len(selected_value_types) == 0 or  # No filter = show all
            (column.value_type and column.value_type in selected_value_types)
        )
        
        # Apply data type filter (if any selected)
        data_type_match = (
            len(selected_data_types) == 0 or  # No filter = show all
            column.type in selected_data_types
        )
        
        if value_type_match and data_type_match:
            filtered_columns.append(column)
    
    return filtered_columns


def extract_unique_types(columns: List[ColumnInsight]) -> tuple[Set[str], Set[str]]:
    """
    Extract unique value types and data types from columns.
    This simulates the logic in ColumnFilter component.
    """
    value_types = set()
    data_types = set()
    
    for column in columns:
        if column.value_type:
            value_types.add(column.value_type)
        data_types.add(column.type)
    
    return value_types, data_types


class TestFrontendFiltering:
    """Test cases for frontend filtering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.columns = [
            ColumnInsight("id", "numeric", "continuous"),
            ColumnInsight("name", "textual", "categorical"),
            ColumnInsight("email", "textual", "categorical"),
            ColumnInsight("age", "numeric", "continuous"),
            ColumnInsight("category", "textual", "categorical"),
            ColumnInsight("timestamp", "temporal", None),
            ColumnInsight("data", "dictionary", None)
        ]
    
    def test_no_filters_shows_all(self):
        """Test that empty filter sets show all columns."""
        result = filter_columns_by_type(self.columns, set(), set())
        assert len(result) == len(self.columns)
        assert all(col in result for col in self.columns)
    
    def test_value_type_filter_only(self):
        """Test filtering by value type only."""
        # Filter for categorical columns
        categorical_filter = {"categorical"}
        result = filter_columns_by_type(self.columns, categorical_filter, set())
        
        expected_names = {"name", "email", "category"}
        result_names = {col.name for col in result}
        assert result_names == expected_names
        assert len(result) == 3
    
    def test_data_type_filter_only(self):
        """Test filtering by data type only."""
        # Filter for textual columns
        textual_filter = {"textual"}
        result = filter_columns_by_type(self.columns, set(), textual_filter)
        
        expected_names = {"name", "email", "category"}
        result_names = {col.name for col in result}
        assert result_names == expected_names
        assert len(result) == 3
    
    def test_combined_filters(self):
        """Test filtering by both value type and data type."""
        # Filter for categorical textual columns
        value_type_filter = {"categorical"}
        data_type_filter = {"textual"}
        result = filter_columns_by_type(self.columns, value_type_filter, data_type_filter)
        
        expected_names = {"name", "email", "category"}
        result_names = {col.name for col in result}
        assert result_names == expected_names
        assert len(result) == 3
    
    def test_restrictive_combined_filters(self):
        """Test restrictive combined filters."""
        # Filter for categorical numeric columns (should be none)
        value_type_filter = {"categorical"}
        data_type_filter = {"numeric"}
        result = filter_columns_by_type(self.columns, value_type_filter, data_type_filter)
        
        assert len(result) == 0
    
    def test_filter_with_null_value_types(self):
        """Test filtering when some columns have null value types."""
        # Filter for columns with value_type (should exclude timestamp, data)
        value_type_filter = {"categorical", "continuous"}
        result = filter_columns_by_type(self.columns, value_type_filter, set())
        
        # Should exclude timestamp and data (which have null value_type)
        result_names = {col.name for col in result}
        assert "timestamp" not in result_names
        assert "data" not in result_names
        assert len(result) == 5  # id, name, email, age, category
    
    def test_multiple_value_types(self):
        """Test filtering with multiple value types selected."""
        value_type_filter = {"categorical", "continuous"}
        result = filter_columns_by_type(self.columns, value_type_filter, set())
        
        # Should include all columns with either categorical or continuous
        expected_names = {"id", "name", "email", "age", "category"}
        result_names = {col.name for col in result}
        assert result_names == expected_names
        assert len(result) == 5
    
    def test_multiple_data_types(self):
        """Test filtering with multiple data types selected."""
        data_type_filter = {"textual", "numeric"}
        result = filter_columns_by_type(self.columns, set(), data_type_filter)
        
        # Should include textual and numeric columns
        expected_names = {"id", "name", "email", "age", "category"}
        result_names = {col.name for col in result}
        assert result_names == expected_names
        assert len(result) == 5
    
    def test_extract_unique_types(self):
        """Test extraction of unique types from columns."""
        value_types, data_types = extract_unique_types(self.columns)
        
        expected_value_types = {"categorical", "continuous"}
        expected_data_types = {"numeric", "textual", "temporal", "dictionary"}
        
        assert value_types == expected_value_types
        assert data_types == expected_data_types
    
    def test_edge_case_empty_columns(self):
        """Test filtering with empty columns list."""
        empty_columns = []
        result = filter_columns_by_type(empty_columns, {"categorical"}, {"textual"})
        assert len(result) == 0
        
        value_types, data_types = extract_unique_types(empty_columns)
        assert len(value_types) == 0
        assert len(data_types) == 0
    
    def test_filter_state_initialization(self):
        """Test the logic for initializing filter state with all types checked."""
        # This simulates the useState initialization in React component
        value_types, data_types = extract_unique_types(self.columns)
        
        # Initialize with all types (default behavior)
        initial_value_types = value_types.copy()
        initial_data_types = data_types.copy()
        
        # Should show all columns when all types are selected
        result = filter_columns_by_type(self.columns, initial_value_types, initial_data_types)
        
        # Note: Columns with null value_type will be filtered out when value type filters are applied
        # Only columns with value_types that are in the filter will show
        columns_with_value_types = [col for col in self.columns if col.value_type is not None]
        assert len(result) == len(columns_with_value_types)
        
        # Test with empty filters (should show all)
        result_empty = filter_columns_by_type(self.columns, set(), set())
        assert len(result_empty) == len(self.columns)
    
    def test_filter_counts_and_badges(self):
        """Test the logic for counting active filters (for badges)."""
        value_type_filter = {"categorical", "continuous"}
        data_type_filter = {"textual"}
        
        # This would be used for badge counts in UI
        value_type_count = len(value_type_filter)
        data_type_count = len(data_type_filter)
        total_filters = value_type_count + data_type_count
        
        assert value_type_count == 2
        assert data_type_count == 1
        assert total_filters == 3
        
        # Test empty filters
        empty_filter = set()
        assert len(empty_filter) == 0
    
    def test_complex_filtering_scenario(self):
        """Test a complex real-world filtering scenario."""
        # Add more diverse columns
        extended_columns = self.columns + [
            ColumnInsight("score", "double", "continuous"),
            ColumnInsight("flag", "boolean", "categorical"),
            ColumnInsight("metadata", "json", None),
            ColumnInsight("created_at", "timestamp", None)
        ]
        
        # Filter for continuous numeric/double columns  
        value_type_filter = {"continuous"}
        data_type_filter = {"numeric", "double"}
        
        result = filter_columns_by_type(extended_columns, value_type_filter, data_type_filter)
        
        # Should include id (numeric, continuous) and score (double, continuous)
        result_names = {col.name for col in result}
        assert result_names == {"id", "age", "score"}
        assert len(result) == 3
