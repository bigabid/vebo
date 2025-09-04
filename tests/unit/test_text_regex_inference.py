"""
Unit tests for TextRegexInference class and pattern complexity analysis.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.text_regex_inference import TextRegexInference, RegexPattern


class TestTextRegexInference:
    """Test cases for TextRegexInference class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tri = TextRegexInference()
    
    def test_pattern_complexity_calculation(self):
        """Test pattern complexity scoring - simpler patterns should have lower scores."""
        # Simple patterns should have lower complexity
        assert self.tri._calculate_pattern_complexity('^[a-z]+$') < self.tri._calculate_pattern_complexity('^[A-Za-z]+$')
        assert self.tri._calculate_pattern_complexity('^[a-z]+$') < self.tri._calculate_pattern_complexity('^[A-Za-z0-9]+$')
        
        # More special characters = higher complexity
        simple = '^abc$'
        complex_chars = '^a[bc]+d{2,3}(xyz|123)*$'
        assert self.tri._calculate_pattern_complexity(simple) < self.tri._calculate_pattern_complexity(complex_chars)
        
        # Longer patterns contribute to complexity  
        short = '^[a-z]$'
        long = '^[a-z]{50}$' * 5  # Very long pattern
        assert self.tri._calculate_pattern_complexity(short) < self.tri._calculate_pattern_complexity(long)
    
    def test_adjust_confidence_for_simplicity(self):
        """Test that simpler patterns get higher confidence when match ratios are similar."""
        # Create patterns with identical match ratios but different complexities
        simple_pattern = RegexPattern(
            pattern='^[a-z]+$',
            description='Simple pattern',
            match_count=10,
            match_ratio=1.0,
            confidence=85.0,
            examples=['abc', 'def']
        )
        
        complex_pattern = RegexPattern(
            pattern='^[A-Za-z0-9_.-]+$',
            description='Complex pattern', 
            match_count=10,
            match_ratio=1.0,
            confidence=85.0,
            examples=['abc', 'def']
        )
        
        patterns = [complex_pattern, simple_pattern]  # Complex first
        adjusted = self.tri._adjust_confidence_for_simplicity(patterns)
        
        # Find the patterns after adjustment
        simple_adjusted = next(p for p in adjusted if p.pattern == '^[a-z]+$')
        complex_adjusted = next(p for p in adjusted if p.pattern == '^[A-Za-z0-9_.-]+$')
        
        # Simple pattern should have higher confidence
        assert simple_adjusted.confidence > complex_adjusted.confidence
    
    def test_infer_patterns_simplicity_preference(self):
        """Test that pattern inference prefers simpler patterns."""
        # Data that matches multiple patterns of different complexity
        data = pd.Series(['abc', 'def', 'xyz', 'hello', 'world'])
        patterns = self.tri.infer_patterns(data, max_patterns=3)
        
        assert len(patterns) > 0
        
        # Should find patterns, and simpler ones should rank higher
        pattern_strings = [p.pattern for p in patterns]
        confidences = [p.confidence for p in patterns]
        
        # Check that we get reasonable patterns
        assert any('[a-z]' in p for p in pattern_strings)
        assert all(c > 0 for c in confidences)
        
        # If we have multiple patterns with similar match ratios,
        # simpler ones should have higher confidence
        if len(patterns) > 1:
            for i in range(len(patterns) - 1):
                if abs(patterns[i].match_ratio - patterns[i+1].match_ratio) < 0.05:
                    # Similar match ratios - confidence should reflect simplicity
                    complexity_i = self.tri._calculate_pattern_complexity(patterns[i].pattern)
                    complexity_next = self.tri._calculate_pattern_complexity(patterns[i+1].pattern)
                    if complexity_i < complexity_next:
                        assert patterns[i].confidence >= patterns[i+1].confidence
    
    def test_common_pattern_detection(self):
        """Test detection of common patterns like emails."""
        # Email data
        emails = pd.Series(['user@example.com', 'admin@test.org', 'support@company.net'])
        patterns = self.tri._check_common_patterns(emails)
        
        email_patterns = [p for p in patterns if 'email' in p.description.lower()]
        assert len(email_patterns) > 0
        assert email_patterns[0].match_ratio > 0.7
        assert email_patterns[0].confidence > 70
    
    def test_deduplicate_patterns(self):
        """Test pattern deduplication."""
        # Create duplicate patterns
        pattern1 = RegexPattern('^[a-z]+$', 'Pattern 1', 10, 1.0, 85.0, ['abc'])
        pattern2 = RegexPattern('^[a-z]+$', 'Pattern 2', 10, 1.0, 90.0, ['def'])  # Same pattern, different details
        pattern3 = RegexPattern('^[0-9]+$', 'Pattern 3', 8, 0.8, 80.0, ['123'])
        
        patterns = [pattern1, pattern2, pattern3]
        unique = self.tri._deduplicate_patterns(patterns)
        
        # Should have only 2 unique patterns (one [a-z] and one [0-9])
        assert len(unique) == 2
        pattern_strings = [p.pattern for p in unique]
        assert '^[a-z]+$' in pattern_strings
        assert '^[0-9]+$' in pattern_strings
    
    def test_empty_series_handling(self):
        """Test handling of empty or null series."""
        empty_series = pd.Series([])
        patterns = self.tri.infer_patterns(empty_series)
        assert patterns == []
        
        null_series = pd.Series([None, None, None])
        patterns = self.tri.infer_patterns(null_series)
        assert patterns == []
    
    def test_mixed_data_pattern_inference(self):
        """Test pattern inference on mixed alphanumeric data."""
        # Product codes or IDs
        mixed_data = pd.Series(['PROD123', 'ITEM456', 'CODE789', 'TEST000'])
        patterns = self.tri.infer_patterns(mixed_data, max_patterns=2)
        
        assert len(patterns) > 0
        assert all(p.match_ratio > 0.5 for p in patterns)  # Should match at least half
        assert all(p.confidence > 50 for p in patterns)   # Reasonable confidence
        
        # Should find patterns that capture some structure (more flexible assertion)
        pattern_strings = [p.pattern for p in patterns]
        assert any(len(p) > 5 for p in pattern_strings)  # Should have some meaningful patterns
        # Check that at least one pattern has reasonable complexity
        assert any('\\w' in p or '[' in p or '+' in p for p in pattern_strings)


class TestRegexPattern:
    """Test cases for RegexPattern dataclass."""
    
    def test_regex_pattern_creation(self):
        """Test creating RegexPattern objects."""
        pattern = RegexPattern(
            pattern='^[a-z]+$',
            description='Lowercase letters',
            match_count=100,
            match_ratio=0.95,
            confidence=88.5,
            examples=['hello', 'world', 'test']
        )
        
        assert pattern.pattern == '^[a-z]+$'
        assert pattern.description == 'Lowercase letters'
        assert pattern.match_count == 100
        assert pattern.match_ratio == 0.95
        assert pattern.confidence == 88.5
        assert pattern.examples == ['hello', 'world', 'test']
