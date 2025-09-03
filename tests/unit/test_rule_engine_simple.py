"""
Simplified unit tests for the RuleEngine class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.rule_engine import (
    RuleEngine, Rule, RuleResult, RuleStatus
)
from vebo_profiler.core.meta_rules import (
    ColumnAttributes, TypeCategory, DiversityLevel, NullabilityLevel
)
from tests.fixtures.sample_data import create_numeric_dataframe, create_textual_dataframe


class TestRuleStatus:
    """Test cases for RuleStatus enum."""
    
    def test_rule_status_values(self):
        """Test that RuleStatus has correct values."""
        assert RuleStatus.PENDING.value == "pending"
        assert RuleStatus.RUNNING.value == "running"
        assert RuleStatus.PASSED.value == "passed"
        assert RuleStatus.FAILED.value == "failed"
        assert RuleStatus.WARNING.value == "warning"
        assert RuleStatus.ERROR.value == "error"
        assert RuleStatus.SKIPPED.value == "skipped"


class TestRuleResult:
    """Test cases for RuleResult class."""
    
    def test_rule_result_creation(self):
        """Test creating a RuleResult object."""
        result = RuleResult(
            rule_id="test_rule",
            rule_name="Test Rule",
            status=RuleStatus.PASSED,
            score=85.5,
            message="Test passed successfully",
            details={"count": 100, "percentage": 85.5},
            execution_time_ms=25.0,
            timestamp="2023-01-01T00:00:00"
        )
        
        assert result.rule_id == "test_rule"
        assert result.rule_name == "Test Rule"
        assert result.status == RuleStatus.PASSED
        assert result.score == 85.5
        assert result.message == "Test passed successfully"
        assert result.details == {"count": 100, "percentage": 85.5}
        assert result.execution_time_ms == 25.0
        assert result.timestamp == "2023-01-01T00:00:00"


class TestRule:
    """Test cases for Rule class."""
    
    def test_rule_creation(self):
        """Test creating a Rule object."""
        rule = Rule(
            id="null_check",
            name="Null Value Check",
            description="Check for null values in column",
            category="data_quality",
            column_types=["numeric", "textual"],
            diversity_levels=["low", "medium", "high"],
            nullability_levels=["low", "medium", "high"],
            requires_cross_column=False,
            dependencies=[],
            code_template="def check_nulls(series):\n    return series.isnull().sum()",
            parameters={}
        )
        
        assert rule.id == "null_check"
        assert rule.name == "Null Value Check"
        assert rule.description == "Check for null values in column"
        assert rule.category == "data_quality"
        assert rule.column_types == ["numeric", "textual"]
        assert rule.diversity_levels == ["low", "medium", "high"]
        assert rule.nullability_levels == ["low", "medium", "high"]
        assert rule.requires_cross_column is False
        assert rule.dependencies == []
        assert rule.code_template == "def check_nulls(series):\n    return series.isnull().sum()"
        assert rule.parameters == {}
    
    def test_rule_creation_with_defaults(self):
        """Test creating a Rule object with default values."""
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"]
        )
        
        assert rule.requires_cross_column is False
        assert rule.dependencies == []
        assert rule.code_template == ""
        assert rule.parameters == {}


class TestRuleEngine:
    """Test cases for RuleEngine class."""
    
    def test_rule_engine_initialization(self):
        """Test RuleEngine initialization."""
        engine = RuleEngine()
        
        assert engine.rules is not None
        assert isinstance(engine.rules, dict)
        assert engine.meta_detector is not None
        assert len(engine.rules) > 0  # Should have built-in rules
    
    def test_get_rule(self):
        """Test getting a rule by ID."""
        engine = RuleEngine()
        
        # Test getting an existing rule
        rule = engine.get_rule("unique_count")
        assert rule is not None
        assert rule.id == "unique_count"
        assert rule.name == "Unique Value Count"
        
        # Test getting a non-existent rule
        rule = engine.get_rule("non_existent_rule")
        assert rule is None
    
    def test_list_rules(self):
        """Test listing all rules."""
        engine = RuleEngine()
        
        rules = engine.list_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0
        
        # Check that all returned items are Rule objects
        for rule in rules:
            assert isinstance(rule, Rule)
            assert hasattr(rule, 'id')
            assert hasattr(rule, 'name')
            assert hasattr(rule, 'category')
    
    def test_get_rules_by_category(self):
        """Test getting rules by category."""
        engine = RuleEngine()
        
        # Test getting rules from a known category
        basic_stats_rules = engine.get_rules_by_category("basic_stats")
        assert isinstance(basic_stats_rules, list)
        assert len(basic_stats_rules) > 0
        
        # Check that all rules are from the correct category
        for rule in basic_stats_rules:
            assert rule.category == "basic_stats"
        
        # Test getting rules from a non-existent category
        non_existent_rules = engine.get_rules_by_category("non_existent_category")
        assert non_existent_rules == []
    
    def test_get_relevant_rules(self):
        """Test getting relevant rules for column attributes."""
        engine = RuleEngine()
        
        # Create column attributes for a numeric column
        numeric_attributes = ColumnAttributes(
            name="age",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=50,
            total_count=100,
            null_count=0,
            most_common_value=30,
            most_common_frequency=5
        )
        
        # Test getting relevant rules
        relevant_rules = engine.get_relevant_rules(numeric_attributes, enable_cross_column=False)
        assert isinstance(relevant_rules, list)
        assert len(relevant_rules) > 0
        
        # Check that all rules are relevant to numeric columns
        for rule in relevant_rules:
            assert "numeric" in rule.column_types or "all" in rule.column_types
    
    def test_get_relevant_rules_with_cross_column(self):
        """Test getting relevant rules including cross-column rules."""
        engine = RuleEngine()
        
        # Create column attributes
        attributes = ColumnAttributes(
            name="age",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=50,
            total_count=100,
            null_count=0,
            most_common_value=30,
            most_common_frequency=5
        )
        
        # Test with cross-column enabled
        relevant_rules = engine.get_relevant_rules(attributes, enable_cross_column=True)
        assert isinstance(relevant_rules, list)
        assert len(relevant_rules) > 0
        
        # Test with cross-column disabled
        relevant_rules_no_cross = engine.get_relevant_rules(attributes, enable_cross_column=False)
        assert isinstance(relevant_rules_no_cross, list)
        assert len(relevant_rules_no_cross) > 0
        
        # Cross-column rules should be filtered out when disabled
        cross_column_rules = [rule for rule in relevant_rules if rule.requires_cross_column]
        cross_column_rules_no_cross = [rule for rule in relevant_rules_no_cross if rule.requires_cross_column]
        
        assert len(cross_column_rules_no_cross) == 0
    
    def test_rule_engine_has_builtin_rules(self):
        """Test that the rule engine has built-in rules loaded."""
        engine = RuleEngine()
        
        # Check for some expected built-in rules
        expected_rules = [
            "unique_count",
            "null_analysis",
            "numeric_stats",
            "most_common_value"
        ]
        
        for rule_id in expected_rules:
            rule = engine.get_rule(rule_id)
            assert rule is not None, f"Expected rule '{rule_id}' not found"
            assert rule.id == rule_id
    
    def test_rule_categories_exist(self):
        """Test that rules are organized by categories."""
        engine = RuleEngine()
        
        # Get all rules and check categories
        all_rules = engine.list_rules()
        categories = set(rule.category for rule in all_rules)
        
        # Should have multiple categories
        assert len(categories) > 1
        
        # Check for expected categories
        expected_categories = ["basic_stats", "numeric_stats", "text_patterns"]
        for category in expected_categories:
            rules_in_category = engine.get_rules_by_category(category)
            assert len(rules_in_category) > 0, f"No rules found in category '{category}'"
    
    def test_rule_engine_with_sample_data(self):
        """Test rule engine with actual sample data."""
        engine = RuleEngine()
        
        # Create simple sample data
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'textual_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Analyze the dataframe to get column attributes
        column_attributes = engine.meta_detector.analyze_dataframe(df)
        
        # Test getting relevant rules for each column
        for col_name, attributes in column_attributes.items():
            relevant_rules = engine.get_relevant_rules(attributes, enable_cross_column=False)
            assert isinstance(relevant_rules, list)
            assert len(relevant_rules) > 0
            
            # Check that rules are relevant to the column type
            for rule in relevant_rules:
                assert attributes.type_category.value in rule.column_types or "all" in rule.column_types


if __name__ == "__main__":
    pytest.main([__file__])
