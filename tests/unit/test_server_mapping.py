"""
Unit tests for server mapping functionality, especially text pattern mapping.
"""

import pytest
import pandas as pd
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from server.main import _map_profiler_to_insights


class TestServerMapping:
    """Test cases for server mapping functionality."""
    
    def test_text_patterns_mapping_new_format(self):
        """Test mapping of text patterns from new format (basic_patterns + inferred_patterns)."""
        # Mock profiler output with new format
        profiler_output = {
            "column_analysis": {
                "test_column": {
                    "data_type": "textual",
                    "null_percentage": 0.1,
                    "checks": [
                        {
                            "check_id": "text_patterns",
                            "rule_id": "text_patterns",
                            "name": "Text Pattern Analysis",
                            "status": "passed",
                            "message": "Text pattern analysis completed. Found 2 inferred patterns.",
                            "details": {
                                "basic_patterns": {
                                    "email_like": {"count": 5, "ratio": 0.5},
                                    "phone_like": {"count": 0, "ratio": 0.0},
                                    "url_like": {"count": 1, "ratio": 0.1}
                                },
                                "inferred_patterns": [
                                    {
                                        "regex": "^[a-z]+@[a-z]+\\.com$",
                                        "description": "Email pattern",
                                        "match_count": 5,
                                        "match_ratio": 0.5,
                                        "confidence": 95.2,
                                        "examples": ["user@test.com", "admin@example.com"]
                                    },
                                    {
                                        "regex": "^[A-Z]{3}\\d{3}$",
                                        "description": "Code pattern",
                                        "match_count": 3,
                                        "match_ratio": 0.3,
                                        "confidence": 88.7,
                                        "examples": ["ABC123", "DEF456"]
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
        
        df = pd.DataFrame({"test_column": ["user@test.com", "admin@example.com", "ABC123"]})
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        # Find the test column
        test_column = next(col for col in insights["columns"] if col["name"] == "test_column")
        
        # Should have textPatterns
        assert "textPatterns" in test_column
        text_patterns = test_column["textPatterns"]
        
        # Check structure
        assert "basic_patterns" in text_patterns
        assert "inferred_patterns" in text_patterns
        assert "status" in text_patterns
        assert "message" in text_patterns
        
        # Check basic patterns
        basic_patterns = text_patterns["basic_patterns"]
        assert "email_like" in basic_patterns
        assert basic_patterns["email_like"]["ratio"] == 0.5
        assert basic_patterns["phone_like"]["ratio"] == 0.0
        assert basic_patterns["url_like"]["ratio"] == 0.1
        
        # Check inferred patterns
        inferred_patterns = text_patterns["inferred_patterns"]
        assert len(inferred_patterns) == 2
        
        email_pattern = inferred_patterns[0]
        assert email_pattern["regex"] == "^[a-z]+@[a-z]+\\.com$"
        assert email_pattern["description"] == "Email pattern"
        assert email_pattern["confidence"] == 95.2
        assert len(email_pattern["examples"]) == 2
    
    def test_text_patterns_mapping_old_format(self):
        """Test mapping of text patterns from old format (patterns only)."""
        # Mock profiler output with old format
        profiler_output = {
            "column_analysis": {
                "test_column": {
                    "data_type": "textual", 
                    "null_percentage": 0.0,
                    "checks": [
                        {
                            "check_id": "text_patterns",
                            "rule_id": "text_patterns",
                            "name": "Text Pattern Analysis",
                            "status": "passed",
                            "message": "Text pattern analysis completed",
                            "details": {
                                "patterns": {
                                    "email_like": {"count": "0", "ratio": 0.0},
                                    "phone_like": {"count": "0", "ratio": 0.0},
                                    "url_like": {"count": "0", "ratio": 0.0}
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        df = pd.DataFrame({"test_column": ["text1", "text2", "text3"]})
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        # Find the test column
        test_column = next(col for col in insights["columns"] if col["name"] == "test_column")
        
        # Should have textPatterns
        assert "textPatterns" in test_column
        text_patterns = test_column["textPatterns"]
        
        # Should map old "patterns" to "basic_patterns"
        assert "basic_patterns" in text_patterns
        assert "inferred_patterns" in text_patterns
        
        # Check that old patterns were mapped correctly
        basic_patterns = text_patterns["basic_patterns"]
        assert "email_like" in basic_patterns
        assert basic_patterns["email_like"]["ratio"] == 0.0
        
        # Should have empty inferred patterns
        assert len(text_patterns["inferred_patterns"]) == 0
    
    def test_no_text_patterns_when_missing(self):
        """Test that textPatterns is not added when text_patterns check is missing."""
        profiler_output = {
            "column_analysis": {
                "test_column": {
                    "data_type": "textual",
                    "null_percentage": 0.0,
                    "checks": [
                        {
                            "check_id": "other_check",
                            "rule_id": "other_check", 
                            "name": "Other Check",
                            "status": "passed",
                            "message": "Other check completed",
                            "details": {"some_data": "value"}
                        }
                    ]
                }
            }
        }
        
        df = pd.DataFrame({"test_column": ["text1", "text2"]})
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        # Find the test column
        test_column = next(col for col in insights["columns"] if col["name"] == "test_column")
        
        # Should not have textPatterns
        assert "textPatterns" not in test_column
    
    def test_no_text_patterns_when_empty(self):
        """Test that textPatterns is not added when patterns are empty."""
        profiler_output = {
            "column_analysis": {
                "test_column": {
                    "data_type": "textual",
                    "null_percentage": 0.0,
                    "checks": [
                        {
                            "check_id": "text_patterns",
                            "rule_id": "text_patterns",
                            "name": "Text Pattern Analysis",
                            "status": "passed",
                            "message": "No patterns found",
                            "details": {}  # Empty details
                        }
                    ]
                }
            }
        }
        
        df = pd.DataFrame({"test_column": ["text1", "text2"]})
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        # Find the test column
        test_column = next(col for col in insights["columns"] if col["name"] == "test_column")
        
        # Should not have textPatterns
        assert "textPatterns" not in test_column
    
    def test_numeric_column_mapping(self):
        """Test that numeric columns are mapped correctly without text patterns."""
        profiler_output = {
            "column_analysis": {
                "numeric_column": {
                    "data_type": "numeric",
                    "null_percentage": 0.0,
                    "checks": [
                        {
                            "check_id": "numeric_stats",
                            "rule_id": "numeric_stats",
                            "name": "Numeric Statistics",
                            "status": "passed",
                            "details": {
                                "statistics": {
                                    "min": "1.0",
                                    "max": "100.0",
                                    "mean": "50.5",
                                    "median": "50.0",
                                    "std": "28.87"
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        df = pd.DataFrame({"numeric_column": [1, 50, 100]})
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        # Find the numeric column
        numeric_column = next(col for col in insights["columns"] if col["name"] == "numeric_column")
        
        # Should have numeric stats but no text patterns
        assert "numeric" in numeric_column
        assert "textPatterns" not in numeric_column
        
        # Check numeric mapping
        numeric_data = numeric_column["numeric"]
        assert numeric_data["min"] == 1.0
        assert numeric_data["max"] == 100.0
        assert numeric_data["avg"] == 50.5
    
    def test_mixed_columns_mapping(self):
        """Test mapping multiple columns with different types."""
        profiler_output = {
            "column_analysis": {
                "text_col": {
                    "data_type": "textual",
                    "null_percentage": 0.0,
                    "checks": [
                        {
                            "check_id": "text_patterns",
                            "rule_id": "text_patterns", 
                            "name": "Text Pattern Analysis",
                            "status": "passed",
                            "details": {
                                "basic_patterns": {
                                    "email_like": {"count": 1, "ratio": 0.5}
                                },
                                "inferred_patterns": []
                            }
                        }
                    ]
                },
                "num_col": {
                    "data_type": "numeric",
                    "null_percentage": 0.1,
                    "checks": []
                }
            }
        }
        
        df = pd.DataFrame({
            "text_col": ["user@test.com", "hello"],
            "num_col": [1, 2]
        })
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        # Should have 2 columns
        assert len(insights["columns"]) == 2
        
        # Find columns
        text_col = next(col for col in insights["columns"] if col["name"] == "text_col")
        num_col = next(col for col in insights["columns"] if col["name"] == "num_col")
        
        # Text column should have patterns
        assert "textPatterns" in text_col
        assert text_col["textPatterns"]["basic_patterns"]["email_like"]["ratio"] == 0.5
        
        # Numeric column should not have patterns
        assert "textPatterns" not in num_col
        assert num_col["type"] == "numeric"
    
    def test_malformed_data_handling(self):
        """Test handling of malformed text pattern data."""
        profiler_output = {
            "column_analysis": {
                "test_column": {
                    "data_type": "textual",
                    "null_percentage": 0.0,
                    "checks": [
                        {
                            "check_id": "text_patterns",
                            "rule_id": "text_patterns",
                            "name": "Text Pattern Analysis", 
                            "status": "passed",
                            "details": "invalid_format"  # Should be dict, not string
                        }
                    ]
                }
            }
        }
        
        df = pd.DataFrame({"test_column": ["text1", "text2"]})
        
        # Should not crash, should handle gracefully
        insights = _map_profiler_to_insights("test_table", {}, df, profiler_output)
        
        test_column = next(col for col in insights["columns"] if col["name"] == "test_column")
        
        # Should not have textPatterns due to malformed data
        assert "textPatterns" not in test_column
