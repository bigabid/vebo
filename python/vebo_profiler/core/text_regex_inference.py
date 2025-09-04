"""
Text regex inference module for automatically detecting patterns in textual data.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import string


@dataclass
class RegexPattern:
    """Represents a detected regex pattern."""
    pattern: str
    description: str
    match_count: int
    match_ratio: float
    confidence: float
    examples: List[str]


class TextRegexInference:
    """
    Class for inferring regex patterns from textual data.
    
    Uses multiple strategies to find meaningful patterns:
    1. Character-based patterns (digits, letters, special chars)
    2. Length-based patterns
    3. Common format patterns (dates, IDs, codes, etc.)
    4. Word-based patterns
    5. Position-based patterns
    """
    
    def __init__(self):
        """Initialize the regex inference engine."""
        self.common_patterns = {
            # Date patterns
            'date_yyyy_mm_dd': (r'^\d{4}-\d{2}-\d{2}$', 'Date in YYYY-MM-DD format'),
            'date_mm_dd_yyyy': (r'^\d{2}/\d{2}/\d{4}$', 'Date in MM/DD/YYYY format'),
            'date_dd_mm_yyyy': (r'^\d{2}/\d{2}/\d{4}$', 'Date in DD/MM/YYYY format'),
            
            # ID patterns
            'numeric_id': (r'^\d+$', 'Numeric identifier'),
            'alphanumeric_id': (r'^[A-Z0-9]+$', 'Alphanumeric identifier'),
            'mixed_case_id': (r'^[A-Za-z0-9]+$', 'Mixed case alphanumeric identifier'),
            
            # Code patterns
            'product_code': (r'^[A-Z]{1,3}\d{3,6}$', 'Product code (letters + numbers)'),
            'license_plate': (r'^[A-Z]{1,3}\s?\d{1,4}$', 'License plate format'),
            
            # Contact patterns
            'email': (r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', 'Email address'),
            'phone_basic': (r'^\d{10}$', '10-digit phone number'),
            'phone_formatted': (r'^\(\d{3}\)\s?\d{3}-\d{4}$', 'Formatted phone number'),
            
            # Address patterns
            'zip_code': (r'^\d{5}(-\d{4})?$', 'US ZIP code'),
            'postal_code': (r'^[A-Z]\d[A-Z]\s?\d[A-Z]\d$', 'Canadian postal code'),
            
            # Financial patterns
            'currency': (r'^\$\d+(\.\d{2})?$', 'Currency amount'),
            'account_number': (r'^\d{8,12}$', 'Account number'),
            
            # Text patterns
            'single_word': (r'^[A-Za-z]+$', 'Single word'),
            'capitalized_word': (r'^[A-Z][a-z]+$', 'Capitalized word'),
            'all_caps': (r'^[A-Z]+$', 'All uppercase letters'),
            'title_case': (r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$', 'Title case text'),
            
            # Special formats
            'version_number': (r'^\d+\.\d+(\.\d+)?$', 'Version number'),
            'ip_address': (r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', 'IP address'),
            'hex_color': (r'^#[0-9A-Fa-f]{6}$', 'Hex color code'),
        }
    
    def infer_patterns(self, series: pd.Series, max_patterns: int = 3) -> List[RegexPattern]:
        """
        Infer regex patterns from a pandas Series of text data.
        
        Args:
            series: Pandas Series containing text data
            max_patterns: Maximum number of patterns to return
            
        Returns:
            List of RegexPattern objects, sorted by confidence
        """
        # Clean the series - remove nulls and convert to string
        clean_series = series.dropna().astype(str)
        
        if len(clean_series) == 0:
            return []
        
        patterns = []
        
        # 1. Check common predefined patterns
        common_patterns = self._check_common_patterns(clean_series)
        patterns.extend(common_patterns)
        
        # 2. Generate character-based patterns
        char_patterns = self._generate_character_patterns(clean_series)
        patterns.extend(char_patterns)
        
        # 3. Generate length-based patterns
        length_patterns = self._generate_length_patterns(clean_series)
        patterns.extend(length_patterns)
        
        # 4. Generate structure-based patterns
        structure_patterns = self._generate_structure_patterns(clean_series)
        patterns.extend(structure_patterns)
        
        # 5. Generate word-based patterns if applicable
        word_patterns = self._generate_word_patterns(clean_series)
        patterns.extend(word_patterns)
        
        # Remove duplicates and adjust confidence based on pattern simplicity
        unique_patterns = self._deduplicate_patterns(patterns)
        patterns_with_adjusted_confidence = self._adjust_confidence_for_simplicity(unique_patterns)
        sorted_patterns = sorted(patterns_with_adjusted_confidence, key=lambda p: p.confidence, reverse=True)
        
        return sorted_patterns[:max_patterns]
    
    def _check_common_patterns(self, series: pd.Series) -> List[RegexPattern]:
        """Check against predefined common patterns."""
        patterns = []
        total_count = len(series)
        
        for pattern_name, (regex, description) in self.common_patterns.items():
            try:
                matches = series.str.match(regex, na=False)
                match_count = matches.sum()
                match_ratio = match_count / total_count
                
                if match_ratio > 0.7:  # High confidence threshold
                    confidence = min(95, 70 + (match_ratio - 0.7) * 83)  # 70-95% confidence
                    examples = series[matches].head(3).tolist()
                    
                    patterns.append(RegexPattern(
                        pattern=regex,
                        description=description,
                        match_count=match_count,
                        match_ratio=match_ratio,
                        confidence=confidence,
                        examples=examples
                    ))
            except Exception:
                # Skip patterns that cause regex errors
                continue
        
        return patterns
    
    def _generate_character_patterns(self, series: pd.Series) -> List[RegexPattern]:
        """Generate patterns based on character composition."""
        patterns = []
        total_count = len(series)
        
        # Analyze character composition of each value
        char_profiles = []
        for value in series:
            profile = self._analyze_character_composition(value)
            char_profiles.append(profile)
        
        # Find common character patterns
        common_compositions = self._find_common_compositions(char_profiles)
        
        for composition, count in common_compositions.items():
            if count / total_count > 0.6:  # At least 60% match
                regex = self._composition_to_regex(composition)
                if regex:
                    match_ratio = count / total_count
                    confidence = min(90, 50 + match_ratio * 40)
                    
                    # Get examples that match this composition
                    examples = []
                    for i, profile in enumerate(char_profiles):
                        if profile == composition and len(examples) < 3:
                            examples.append(series.iloc[i])
                    
                    patterns.append(RegexPattern(
                        pattern=regex,
                        description=f"Pattern: {composition}",
                        match_count=count,
                        match_ratio=match_ratio,
                        confidence=confidence,
                        examples=examples
                    ))
        
        return patterns
    
    def _generate_length_patterns(self, series: pd.Series) -> List[RegexPattern]:
        """Generate patterns based on string length."""
        patterns = []
        total_count = len(series)
        
        # Analyze length distribution
        lengths = series.str.len()
        length_counts = lengths.value_counts()
        
        # Check for fixed-length patterns
        dominant_length = length_counts.index[0]
        dominant_count = length_counts.iloc[0]
        
        if dominant_count / total_count > 0.8:  # 80% have the same length
            # Create a pattern based on the most common character types at each position
            same_length_values = series[lengths == dominant_length]
            positional_pattern = self._analyze_positional_patterns(same_length_values)
            
            if positional_pattern:
                regex = '^' + positional_pattern + '$'
                confidence = min(85, 60 + (dominant_count / total_count - 0.8) * 125)
                
                patterns.append(RegexPattern(
                    pattern=regex,
                    description=f"Fixed length ({dominant_length} characters) with positional pattern",
                    match_count=dominant_count,
                    match_ratio=dominant_count / total_count,
                    confidence=confidence,
                    examples=same_length_values.head(3).tolist()
                ))
        
        return patterns
    
    def _generate_structure_patterns(self, series: pd.Series) -> List[RegexPattern]:
        """Generate patterns based on structural elements like separators."""
        patterns = []
        total_count = len(series)
        
        # Check for common separators
        separators = ['-', '/', '_', '.', ' ', ':', ';', ',']
        
        for sep in separators:
            if series.str.contains(re.escape(sep), na=False).sum() / total_count > 0.7:
                # Analyze the structure around this separator
                structured_pattern = self._analyze_separator_structure(series, sep)
                if structured_pattern:
                    # Count how many values match this pattern
                    try:
                        matches = series.str.match(structured_pattern, na=False).sum()
                        match_ratio = matches / total_count
                        
                        if match_ratio > 0.6:
                            confidence = min(80, 40 + match_ratio * 50)
                            examples = series[series.str.match(structured_pattern, na=False)].head(3).tolist()
                            
                            patterns.append(RegexPattern(
                                pattern=structured_pattern,
                                description=f"Structured pattern with '{sep}' separator",
                                match_count=matches,
                                match_ratio=match_ratio,
                                confidence=confidence,
                                examples=examples
                            ))
                    except Exception:
                        continue
        
        return patterns
    
    def _generate_word_patterns(self, series: pd.Series) -> List[RegexPattern]:
        """Generate patterns for word-based text."""
        patterns = []
        total_count = len(series)
        
        # Check if values are primarily word-based
        word_based_count = series.str.match(r'^[A-Za-z\s]+$', na=False).sum()
        
        if word_based_count / total_count > 0.5:
            # Analyze word count patterns
            word_counts = series.str.split().str.len()
            common_word_count = word_counts.mode().iloc[0] if not word_counts.mode().empty else 1
            same_word_count = (word_counts == common_word_count).sum()
            
            if same_word_count / total_count > 0.7:
                if common_word_count == 1:
                    pattern = r'^[A-Za-z]+$'
                    description = "Single word"
                elif common_word_count == 2:
                    pattern = r'^[A-Za-z]+\s[A-Za-z]+$'
                    description = "Two words"
                else:
                    word_part = r'[A-Za-z]+\s'
                    pattern = f'^({word_part}{{{common_word_count-1}}}[A-Za-z]+)$'
                    description = f"{common_word_count} words"
                
                confidence = min(75, 50 + (same_word_count / total_count - 0.7) * 83)
                examples = series[word_counts == common_word_count].head(3).tolist()
                
                patterns.append(RegexPattern(
                    pattern=pattern,
                    description=description,
                    match_count=same_word_count,
                    match_ratio=same_word_count / total_count,
                    confidence=confidence,
                    examples=examples
                ))
        
        return patterns
    
    def _analyze_character_composition(self, value: str) -> str:
        """Analyze the character composition of a string."""
        if not value:
            return "empty"
        
        composition = []
        
        # Count different character types
        digits = sum(c.isdigit() for c in value)
        letters = sum(c.isalpha() for c in value)
        uppers = sum(c.isupper() for c in value)
        lowers = sum(c.islower() for c in value)
        spaces = sum(c.isspace() for c in value)
        specials = len(value) - digits - letters - spaces
        
        total_len = len(value)
        
        # Create a composition signature
        if digits == total_len:
            return "all_digits"
        elif letters == total_len:
            if uppers == total_len:
                return "all_uppercase"
            elif lowers == total_len:
                return "all_lowercase"
            elif uppers == 1 and lowers == total_len - 1:
                return "title_case"
            else:
                return "mixed_case_letters"
        elif digits > 0 and letters > 0 and specials == 0:
            return "alphanumeric"
        elif digits > 0 and letters > 0 and specials > 0:
            return "mixed_with_specials"
        else:
            return "other"
    
    def _find_common_compositions(self, profiles: List[str]) -> Dict[str, int]:
        """Find the most common character compositions."""
        return Counter(profiles)
    
    def _composition_to_regex(self, composition: str) -> Optional[str]:
        """Convert a character composition to a regex pattern."""
        patterns = {
            "all_digits": r'^\d+$',
            "all_uppercase": r'^[A-Z]+$',
            "all_lowercase": r'^[a-z]+$',
            "title_case": r'^[A-Z][a-z]*$',
            "mixed_case_letters": r'^[A-Za-z]+$',
            "alphanumeric": r'^[A-Za-z0-9]+$',
        }
        return patterns.get(composition)
    
    def _analyze_positional_patterns(self, series: pd.Series) -> Optional[str]:
        """Analyze character patterns at each position for same-length strings."""
        if len(series) == 0:
            return None
        
        # Get the common length
        length = series.str.len().iloc[0]
        pattern_parts = []
        
        for pos in range(length):
            # Get characters at this position
            chars_at_pos = series.str[pos].tolist()
            
            # Analyze what type of characters appear here
            digits = sum(1 for c in chars_at_pos if c.isdigit())
            uppers = sum(1 for c in chars_at_pos if c.isupper())
            lowers = sum(1 for c in chars_at_pos if c.islower())
            spaces = sum(1 for c in chars_at_pos if c.isspace())
            specials = len(chars_at_pos) - digits - uppers - lowers - spaces
            
            total = len(chars_at_pos)
            
            # Determine the pattern for this position
            if digits / total > 0.8:
                pattern_parts.append(r'\d')
            elif uppers / total > 0.8:
                pattern_parts.append(r'[A-Z]')
            elif lowers / total > 0.8:
                pattern_parts.append(r'[a-z]')
            elif (uppers + lowers) / total > 0.8:
                pattern_parts.append(r'[A-Za-z]')
            elif (digits + uppers + lowers) / total > 0.8:
                pattern_parts.append(r'[A-Za-z0-9]')
            elif spaces / total > 0.8:
                pattern_parts.append(r'\s')
            else:
                # Too diverse, use a more general pattern
                unique_chars = set(chars_at_pos)
                if len(unique_chars) <= 3:
                    escaped_chars = [re.escape(c) for c in unique_chars]
                    pattern_parts.append(f"[{''.join(escaped_chars)}]")
                else:
                    pattern_parts.append(r'.')
        
        return ''.join(pattern_parts)
    
    def _analyze_separator_structure(self, series: pd.Series, separator: str) -> Optional[str]:
        """Analyze the structure around a separator."""
        # Split by separator and analyze each part
        split_series = series.str.split(re.escape(separator), expand=True)
        
        if split_series.shape[1] < 2:
            return None
        
        # Analyze the pattern of each part
        parts_patterns = []
        for col in range(split_series.shape[1]):
            part_series = split_series[col].dropna()
            if len(part_series) == 0:
                continue
            
            # Find common pattern for this part
            part_pattern = self._find_part_pattern(part_series)
            if part_pattern:
                parts_patterns.append(part_pattern)
        
        if len(parts_patterns) >= 2:
            escaped_sep = re.escape(separator)
            return f"^{escaped_sep.join(parts_patterns)}$"
        
        return None
    
    def _find_part_pattern(self, part_series: pd.Series) -> Optional[str]:
        """Find a pattern for a part of a structured string."""
        if len(part_series) == 0:
            return None
        
        # Check if all parts have the same length
        lengths = part_series.str.len()
        if lengths.nunique() == 1:
            common_length = lengths.iloc[0]
            if common_length <= 10:  # Only for reasonable lengths
                pos_pattern = self._analyze_positional_patterns(part_series)
                if pos_pattern:
                    return pos_pattern
        
        # Check for common character types
        all_digits = part_series.str.match(r'^\d+$', na=False).all()
        all_letters = part_series.str.match(r'^[A-Za-z]+$', na=False).all()
        all_upper = part_series.str.match(r'^[A-Z]+$', na=False).all()
        all_lower = part_series.str.match(r'^[a-z]+$', na=False).all()
        
        if all_digits:
            return r'\d+'
        elif all_upper:
            return r'[A-Z]+'
        elif all_lower:
            return r'[a-z]+'
        elif all_letters:
            return r'[A-Za-z]+'
        else:
            return r'[A-Za-z0-9]+'
    
    def _deduplicate_patterns(self, patterns: List[RegexPattern]) -> List[RegexPattern]:
        """Remove duplicate or overly similar patterns."""
        if not patterns:
            return []
        
        unique_patterns = []
        seen_patterns = set()
        
        for pattern in patterns:
            # Use pattern string as key for deduplication
            if pattern.pattern not in seen_patterns:
                seen_patterns.add(pattern.pattern)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _calculate_pattern_complexity(self, pattern: str) -> int:
        """
        Calculate pattern complexity score. Lower score = simpler pattern.
        
        Args:
            pattern: Regex pattern string
            
        Returns:
            Complexity score (lower is simpler)
        """
        complexity = 0
        
        # Count special regex characters
        special_chars = ['+', '*', '?', '{', '}', '[', ']', '(', ')', '|', '^', '$', '.', '\\']
        for char in special_chars:
            complexity += pattern.count(char)
        
        # Length contributes to complexity
        complexity += len(pattern) // 10  # Every 10 characters adds 1 to complexity
        
        # Character classes are simpler than explicit lists
        if '[a-z]' in pattern.lower():
            complexity -= 2
        if '[a-za-z]' in pattern.lower():
            complexity -= 1
        if '[0-9]' in pattern or '\\d' in pattern:
            complexity -= 2
            
        return max(0, complexity)
    
    def _adjust_confidence_for_simplicity(self, patterns: List[RegexPattern]) -> List[RegexPattern]:
        """
        Adjust pattern confidence to prefer simpler patterns when match ratios are similar.
        
        Args:
            patterns: List of regex patterns
            
        Returns:
            List of patterns with adjusted confidence scores
        """
        if not patterns:
            return patterns
        
        # Group patterns by similar match ratios (within 0.05 difference)
        ratio_groups = {}
        for pattern in patterns:
            ratio_key = round(pattern.match_ratio * 20) / 20  # Round to nearest 0.05
            if ratio_key not in ratio_groups:
                ratio_groups[ratio_key] = []
            ratio_groups[ratio_key].append(pattern)
        
        adjusted_patterns = []
        for ratio_key, group in ratio_groups.items():
            if len(group) > 1:
                # Multiple patterns with similar match ratios - adjust for simplicity
                complexities = [(pattern, self._calculate_pattern_complexity(pattern.pattern)) for pattern in group]
                min_complexity = min(complexity for _, complexity in complexities)
                
                for pattern, complexity in complexities:
                    # Boost confidence for simpler patterns
                    simplicity_bonus = max(0, (min_complexity + 3 - complexity) * 2)  # Up to 6 point bonus
                    new_confidence = min(99.5, pattern.confidence + simplicity_bonus)
                    
                    adjusted_patterns.append(RegexPattern(
                        pattern=pattern.pattern,
                        description=pattern.description,
                        match_count=pattern.match_count,
                        match_ratio=pattern.match_ratio,
                        confidence=new_confidence,
                        examples=pattern.examples
                    ))
            else:
                # Single pattern in this ratio group - keep as is
                adjusted_patterns.extend(group)
        
        return adjusted_patterns
