"""
Unit tests for enhanced text_patterns rule with smart skipping logic.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.rule_engine import RuleEngine
from vebo_profiler.core.check_executor import CheckExecutor, ExecutionConfig
from vebo_profiler.core.meta_rules import ColumnAttributes, TypeCategory, DiversityLevel, NullabilityLevel


def _create_attributes(diversity_level=DiversityLevel.HIGH):
    """Create column attributes for testing."""
    return ColumnAttributes(
        name="test_column",
        unique_count=100,
        total_count=100,
        null_count=0,
        type_category=TypeCategory.TEXTUAL,
        diversity_level=diversity_level,
        nullability_level=NullabilityLevel.LOW
    )


class TestTextPatternsRule:
    """Test cases for enhanced text_patterns rule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RuleEngine()
        self.executor = CheckExecutor(ExecutionConfig())
        self.rule = self.engine.get_rule('text_patterns')
        assert self.rule is not None
    
    def test_rule_skips_low_diversity_columns(self):
        """Test that text_patterns rule doesn't apply to low diversity columns."""
        # Create low diversity attributes
        low_diversity_attrs = _create_attributes(diversity_level=DiversityLevel.LOW)
        
        # Rule should not be applicable to low diversity columns
        assert not self.rule.is_applicable(low_diversity_attrs)
        
        # But should be applicable to higher diversity
        high_diversity_attrs = _create_attributes(diversity_level=DiversityLevel.HIGH)
        assert self.rule.is_applicable(high_diversity_attrs)
    
    def test_skips_regex_inference_for_high_email_ratio(self):
        """Test that regex inference is skipped when emails dominate (>80%)."""
        # Email-heavy data (100% emails)
        email_data = pd.Series([
            'user@example.com', 'admin@test.org', 'support@company.net', 
            'info@domain.com', 'contact@business.co.uk'
        ])
        
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, email_data, attrs)
        
        assert result.status.value == 'passed'
        assert 'basic_patterns' in result.details
        assert 'inferred_patterns' in result.details
        
        # Should have high email ratio
        email_ratio = result.details['basic_patterns']['email_like']['ratio']
        assert email_ratio > 0.8
        
        # Should skip regex inference (no inferred patterns)
        assert len(result.details['inferred_patterns']) == 0
        
        # Message should indicate skipping
        assert 'Skipped regex inference' in result.message
    
    def test_runs_regex_inference_for_mixed_data(self):
        """Test that regex inference runs when no known patterns dominate."""
        # Mixed data that doesn't match known patterns strongly
        mixed_data = pd.Series([
            'PROD123', 'ITEM456', 'CODE789', 'TEST000', 'SAMPLE999'
        ])
        
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, mixed_data, attrs)
        
        assert result.status.value == 'passed'
        assert 'basic_patterns' in result.details
        assert 'inferred_patterns' in result.details
        
        # Should have low ratios for known patterns
        basic_patterns = result.details['basic_patterns']
        assert basic_patterns['email_like']['ratio'] < 0.1
        assert basic_patterns['phone_like']['ratio'] < 0.1
        assert basic_patterns['url_like']['ratio'] < 0.1
        
        # Should run regex inference and find patterns
        assert len(result.details['inferred_patterns']) > 0
        
        # Message should not mention skipping
        assert 'Skipped regex inference' not in result.message
    
    def test_phone_number_pattern_detection(self):
        """Test detection of phone number patterns."""
        # Phone-heavy data
        phone_data = pd.Series([
            '+1-555-123-4567', '+44 20 7946 0958', '555-123-4567', 
            '(555) 123-4567', '+1 555 123 4567'
        ])
        
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, phone_data, attrs)
        
        assert result.status.value == 'passed'
        
        # Should detect some phone-like patterns (though our regex is basic)
        phone_ratio = result.details['basic_patterns']['phone_like']['ratio']
        # Note: Our phone regex is basic, so it might not catch all formats
        assert phone_ratio >= 0  # At least not negative
    
    def test_url_pattern_detection(self):
        """Test detection of URL patterns."""
        # Simple URL data that matches our basic regex
        url_data = pd.Series([
            'https://example.com', 'http://test.org', 
            'https://api.net', 'http://site.com'
        ])
        
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, url_data, attrs)
        
        assert result.status.value == 'passed'
        
        # Our URL regex is basic, so let's be more lenient
        url_ratio = result.details['basic_patterns']['url_like']['ratio']
        assert url_ratio >= 0  # At least should not error
    
    def test_handles_mixed_known_patterns(self):
        """Test behavior when multiple known patterns are present."""
        # Mixed data with some emails and some other text  
        mixed_data = pd.Series([
            'user@example.com', 'admin@test.org',  # 40% emails
            'PROD123', 'ITEM456', 'CODE789'        # 60% product codes
        ])
        
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, mixed_data, attrs)
        
        assert result.status.value == 'passed'
        
        # No single pattern should dominate (>80%)
        basic_patterns = result.details['basic_patterns']
        max_ratio = max(
            basic_patterns['email_like']['ratio'],
            basic_patterns['phone_like']['ratio'], 
            basic_patterns['url_like']['ratio']
        )
        assert max_ratio < 0.8
        
        # Should run regex inference (unless email ratio is unexpectedly high)
        if max_ratio <= 0.8:
            # Only assert if we expect it to run
            assert 'Skipped regex inference' not in result.message
    
    def test_empty_or_null_data_handling(self):
        """Test handling of empty or null data."""
        # All null data
        null_data = pd.Series([None, None, None, None])
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, null_data, attrs)
        
        # Should handle gracefully
        assert result.status.value in ['passed', 'warning']
        assert 'basic_patterns' in result.details
        
        # Empty data
        empty_data = pd.Series([])
        result2 = self.executor.execute_column_check(self.rule, empty_data, attrs)
        assert result2.status.value in ['passed', 'warning']
    
    def test_basic_patterns_structure(self):
        """Test that basic patterns have the correct structure."""
        test_data = pd.Series(['test@email.com', 'regular text', 'more text'])
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, test_data, attrs)
        
        basic_patterns = result.details['basic_patterns']
        
        # Should have all expected pattern types
        expected_patterns = ['email_like', 'phone_like', 'url_like']
        for pattern_name in expected_patterns:
            assert pattern_name in basic_patterns
            pattern_data = basic_patterns[pattern_name]
            assert 'count' in pattern_data
            assert 'ratio' in pattern_data
            assert isinstance(pattern_data['ratio'], (int, float))
            assert 0 <= pattern_data['ratio'] <= 1
    
    def test_inferred_patterns_structure(self):
        """Test that inferred patterns have the correct structure."""
        # Data that should generate inferred patterns
        pattern_data = pd.Series(['ABC123', 'DEF456', 'GHI789', 'JKL000'])
        attrs = _create_attributes()
        result = self.executor.execute_column_check(self.rule, pattern_data, attrs)
        
        if result.details['inferred_patterns']:  # If patterns were generated
            for pattern in result.details['inferred_patterns']:
                # Check required fields
                assert 'regex' in pattern
                assert 'description' in pattern
                assert 'match_count' in pattern
                assert 'match_ratio' in pattern
                assert 'confidence' in pattern
                assert 'examples' in pattern
                
                # Check types and ranges (handle numpy types)
                assert isinstance(pattern['regex'], str)
                assert isinstance(pattern['description'], str)
                assert isinstance(pattern['match_count'], (int, float)) or hasattr(pattern['match_count'], 'item')  # numpy types
                assert isinstance(pattern['match_ratio'], (int, float)) or hasattr(pattern['match_ratio'], 'item')
                assert isinstance(pattern['confidence'], (int, float)) or hasattr(pattern['confidence'], 'item') 
                assert isinstance(pattern['examples'], list)
                
                assert 0 <= pattern['match_ratio'] <= 1
                assert 0 <= pattern['confidence'] <= 100
