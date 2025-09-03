"""
Unit tests for the RuleEngine class.
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

from vebo_profiler.core.rule_engine import (
    RuleEngine, Rule, RuleResult, RuleStatus, 
    RuleCategory, RulePriority, RuleComplexity
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
    
    def test_rule_result_to_dict(self):
        """Test converting RuleResult to dictionary."""
        result = RuleResult(
            rule_id="test_rule",
            rule_name="Test Rule",
            status=RuleStatus.PASSED,
            score=85.5,
            message="Test passed successfully",
            details={"count": 100},
            execution_time_ms=25.0,
            timestamp="2023-01-01T00:00:00"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["rule_id"] == "test_rule"
        assert result_dict["rule_name"] == "Test Rule"
        assert result_dict["status"] == "passed"
        assert result_dict["score"] == 85.5
        assert result_dict["message"] == "Test passed successfully"
        assert result_dict["details"] == {"count": 100}
        assert result_dict["execution_time_ms"] == 25.0
        assert result_dict["timestamp"] == "2023-01-01T00:00:00"


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
            priority=RulePriority.HIGH,
            complexity=RuleComplexity.LOW,
            code_template="def check_nulls(series):\n    return series.isnull().sum()",
            parameters={},
            enabled=True
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
        assert rule.priority == RulePriority.HIGH
        assert rule.complexity == RuleComplexity.LOW
        assert rule.enabled is True
    
    def test_rule_is_applicable(self):
        """Test rule applicability checking."""
        rule = Rule(
            id="numeric_check",
            name="Numeric Check",
            description="Check for numeric values",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium", "high"],
            nullability_levels=["low", "medium"],
            requires_cross_column=False
        )
        
        # Test applicable case
        attributes = ColumnAttributes(
            name="age",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=50,
            total_count=100,
            null_count=0
        )
        
        assert rule.is_applicable(attributes) is True
        
        # Test non-applicable case (wrong type)
        attributes.type_category = TypeCategory.TEXTUAL
        assert rule.is_applicable(attributes) is False
        
        # Test non-applicable case (wrong diversity)
        attributes.type_category = TypeCategory.NUMERIC
        attributes.diversity_level = DiversityLevel.CONSTANT
        assert rule.is_applicable(attributes) is False
        
        # Test non-applicable case (wrong nullability)
        attributes.diversity_level = DiversityLevel.MEDIUM
        attributes.nullability_level = NullabilityLevel.HIGH
        assert rule.is_applicable(attributes) is False


class TestRuleEngine:
    """Test cases for RuleEngine class."""
    
    def test_rule_engine_initialization(self):
        """Test RuleEngine initialization."""
        engine = RuleEngine()
        
        assert engine.rules == {}
        assert engine.rule_categories == {}
        assert engine.rule_dependencies == {}
        assert engine.rule_priorities == {}
        assert engine.rule_complexities == {}
    
    def test_add_rule(self):
        """Test adding a rule to the engine."""
        engine = RuleEngine()
        
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False
        )
        
        engine.add_rule(rule)
        
        assert "test_rule" in engine.rules
        assert engine.rules["test_rule"] == rule
        assert "data_quality" in engine.rule_categories
        assert "test_rule" in engine.rule_categories["data_quality"]
    
    def test_remove_rule(self):
        """Test removing a rule from the engine."""
        engine = RuleEngine()
        
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False
        )
        
        engine.add_rule(rule)
        assert "test_rule" in engine.rules
        
        engine.remove_rule("test_rule")
        assert "test_rule" not in engine.rules
        assert "test_rule" not in engine.rule_categories["data_quality"]
    
    def test_get_rule(self):
        """Test getting a rule by ID."""
        engine = RuleEngine()
        
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False
        )
        
        engine.add_rule(rule)
        
        retrieved_rule = engine.get_rule("test_rule")
        assert retrieved_rule == rule
        
        # Test non-existent rule
        assert engine.get_rule("non_existent") is None
    
    def test_list_rules(self):
        """Test listing all rules."""
        engine = RuleEngine()
        
        rule1 = Rule(
            id="rule1",
            name="Rule 1",
            description="First rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False
        )
        
        rule2 = Rule(
            id="rule2",
            name="Rule 2",
            description="Second rule",
            category="data_quality",
            column_types=["textual"],
            diversity_levels=["high"],
            nullability_levels=["medium"],
            requires_cross_column=False
        )
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        rules = engine.list_rules()
        assert len(rules) == 2
        assert rule1 in rules
        assert rule2 in rules
    
    def test_get_rules_by_category(self):
        """Test getting rules by category."""
        engine = RuleEngine()
        
        rule1 = Rule(
            id="rule1",
            name="Rule 1",
            description="First rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False
        )
        
        rule2 = Rule(
            id="rule2",
            name="Rule 2",
            description="Second rule",
            category="data_types",
            column_types=["textual"],
            diversity_levels=["high"],
            nullability_levels=["medium"],
            requires_cross_column=False
        )
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        data_quality_rules = engine.get_rules_by_category("data_quality")
        assert len(data_quality_rules) == 1
        assert rule1 in data_quality_rules
        
        data_types_rules = engine.get_rules_by_category("data_types")
        assert len(data_types_rules) == 1
        assert rule2 in data_types_rules
        
        # Test non-existent category
        assert engine.get_rules_by_category("non_existent") == []
    
    def test_get_relevant_rules(self):
        """Test getting relevant rules for column attributes."""
        engine = RuleEngine()
        
        # Add rules for different scenarios
        numeric_rule = Rule(
            id="numeric_rule",
            name="Numeric Rule",
            description="Rule for numeric columns",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium", "high"],
            nullability_levels=["low", "medium"],
            requires_cross_column=False
        )
        
        textual_rule = Rule(
            id="textual_rule",
            name="Textual Rule",
            description="Rule for textual columns",
            category="data_quality",
            column_types=["textual"],
            diversity_levels=["high"],
            nullability_levels=["low"],
            requires_cross_column=False
        )
        
        cross_column_rule = Rule(
            id="cross_column_rule",
            name="Cross Column Rule",
            description="Rule requiring cross-column analysis",
            category="cross_column",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=True
        )
        
        engine.add_rule(numeric_rule)
        engine.add_rule(textual_rule)
        engine.add_rule(cross_column_rule)
        
        # Test with numeric column attributes
        numeric_attributes = ColumnAttributes(
            name="age",
            type_category=TypeCategory.NUMERIC,
            diversity_level=DiversityLevel.MEDIUM,
            nullability_level=NullabilityLevel.LOW,
            unique_count=50,
            total_count=100,
            null_count=0
        )
        
        relevant_rules = engine.get_relevant_rules(numeric_attributes, enable_cross_column=False)
        assert len(relevant_rules) == 1
        assert numeric_rule in relevant_rules
        
        # Test with cross-column enabled
        relevant_rules = engine.get_relevant_rules(numeric_attributes, enable_cross_column=True)
        assert len(relevant_rules) == 2
        assert numeric_rule in relevant_rules
        assert cross_column_rule in relevant_rules
        
        # Test with textual column attributes
        textual_attributes = ColumnAttributes(
            name="name",
            type_category=TypeCategory.TEXTUAL,
            diversity_level=DiversityLevel.HIGH,
            nullability_level=NullabilityLevel.LOW,
            unique_count=100,
            total_count=100,
            null_count=0
        )
        
        relevant_rules = engine.get_relevant_rules(textual_attributes, enable_cross_column=False)
        assert len(relevant_rules) == 1
        assert textual_rule in relevant_rules
    
    def test_get_rule_dependencies(self):
        """Test getting rule dependencies."""
        engine = RuleEngine()
        
        rule1 = Rule(
            id="rule1",
            name="Rule 1",
            description="First rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=[]
        )
        
        rule2 = Rule(
            id="rule2",
            name="Rule 2",
            description="Second rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=["rule1"]
        )
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        dependencies = engine.get_rule_dependencies("rule2")
        assert dependencies == ["rule1"]
        
        # Test rule with no dependencies
        dependencies = engine.get_rule_dependencies("rule1")
        assert dependencies == []
        
        # Test non-existent rule
        dependencies = engine.get_rule_dependencies("non_existent")
        assert dependencies == []
    
    def test_validate_rule_dependencies(self):
        """Test validating rule dependencies."""
        engine = RuleEngine()
        
        # Add rules with dependencies
        rule1 = Rule(
            id="rule1",
            name="Rule 1",
            description="First rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=[]
        )
        
        rule2 = Rule(
            id="rule2",
            name="Rule 2",
            description="Second rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=["rule1"]
        )
        
        rule3 = Rule(
            id="rule3",
            name="Rule 3",
            description="Third rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=["rule1", "rule2"]
        )
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        engine.add_rule(rule3)
        
        # Test valid dependencies
        assert engine.validate_rule_dependencies("rule2") is True
        assert engine.validate_rule_dependencies("rule3") is True
        
        # Test invalid dependencies (circular dependency)
        rule4 = Rule(
            id="rule4",
            name="Rule 4",
            description="Fourth rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=["rule4"]  # Self-dependency
        )
        
        engine.add_rule(rule4)
        assert engine.validate_rule_dependencies("rule4") is False
    
    def test_load_rules_from_file(self):
        """Test loading rules from a file."""
        engine = RuleEngine()
        
        # Create a mock rules file
        rules_data = {
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule",
                    "description": "A test rule",
                    "category": "data_quality",
                    "column_types": ["numeric"],
                    "diversity_levels": ["medium"],
                    "nullability_levels": ["low"],
                    "requires_cross_column": False,
                    "dependencies": [],
                    "priority": "high",
                    "complexity": "low",
                    "code_template": "def check(series): return True",
                    "parameters": {},
                    "enabled": True
                }
            ]
        }
        
        with patch('builtins.open', mock_open(json.dumps(rules_data))):
            engine.load_rules_from_file("test_rules.json")
            
            assert "test_rule" in engine.rules
            rule = engine.rules["test_rule"]
            assert rule.name == "Test Rule"
            assert rule.category == "data_quality"
    
    def test_save_rules_to_file(self):
        """Test saving rules to a file."""
        engine = RuleEngine()
        
        rule = Rule(
            id="test_rule",
            name="Test Rule",
            description="A test rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            dependencies=[],
            priority=RulePriority.HIGH,
            complexity=RuleComplexity.LOW,
            code_template="def check(series): return True",
            parameters={},
            enabled=True
        )
        
        engine.add_rule(rule)
        
        with patch('builtins.open', mock_open()) as mock_file:
            engine.save_rules_to_file("test_rules.json")
            
            mock_file.assert_called_once_with("test_rules.json", 'w')
            mock_file().write.assert_called_once()
    
    def test_get_rule_statistics(self):
        """Test getting rule statistics."""
        engine = RuleEngine()
        
        # Add rules of different categories and priorities
        rule1 = Rule(
            id="rule1",
            name="Rule 1",
            description="First rule",
            category="data_quality",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=False,
            priority=RulePriority.HIGH,
            complexity=RuleComplexity.LOW
        )
        
        rule2 = Rule(
            id="rule2",
            name="Rule 2",
            description="Second rule",
            category="data_types",
            column_types=["textual"],
            diversity_levels=["high"],
            nullability_levels=["medium"],
            requires_cross_column=False,
            priority=RulePriority.MEDIUM,
            complexity=RuleComplexity.HIGH
        )
        
        rule3 = Rule(
            id="rule3",
            name="Rule 3",
            description="Third rule",
            category="cross_column",
            column_types=["numeric"],
            diversity_levels=["medium"],
            nullability_levels=["low"],
            requires_cross_column=True,
            priority=RulePriority.LOW,
            complexity=RuleComplexity.MEDIUM
        )
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        engine.add_rule(rule3)
        
        stats = engine.get_rule_statistics()
        
        assert stats["total_rules"] == 3
        assert stats["categories"]["data_quality"] == 1
        assert stats["categories"]["data_types"] == 1
        assert stats["categories"]["cross_column"] == 1
        assert stats["priorities"]["high"] == 1
        assert stats["priorities"]["medium"] == 1
        assert stats["priorities"]["low"] == 1
        assert stats["complexities"]["low"] == 1
        assert stats["complexities"]["medium"] == 1
        assert stats["complexities"]["high"] == 1
        assert stats["cross_column_rules"] == 1


def mock_open(content=""):
    """Mock open function for testing file operations."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open(read_data=content)


if __name__ == "__main__":
    pytest.main([__file__])
