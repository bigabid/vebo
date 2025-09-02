"""
Rule engine for managing and executing data profiling rules.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .meta_rules import ColumnAttributes, MetaRuleDetector


class RuleStatus(Enum):
    """Status of a rule execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class RuleResult:
    """Result of a rule execution."""
    rule_id: str
    rule_name: str
    status: RuleStatus
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    execution_time_ms: float
    timestamp: str


@dataclass
class Rule:
    """Definition of a data profiling rule."""
    id: str
    name: str
    description: str
    category: str
    column_types: List[str]  # Which column types this rule applies to
    diversity_levels: List[str]  # Which diversity levels this rule applies to
    nullability_levels: List[str]  # Which nullability levels this rule applies to
    requires_cross_column: bool = False
    dependencies: List[str] = None  # Other rules this depends on
    code_template: str = ""
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}


class RuleEngine:
    """
    Engine for managing and executing data profiling rules.
    """
    
    def __init__(self):
        """Initialize the rule engine."""
        self.rules: Dict[str, Rule] = {}
        self.meta_detector = MetaRuleDetector()
        self._load_builtin_rules()
    
    def _load_builtin_rules(self):
        """Load built-in rules based on the rule categories."""
        self._add_basic_stats_rules()
        self._add_numeric_rules()
        self._add_textual_rules()
        self._add_temporal_rules()
        self._add_cross_column_rules()
    
    def _add_basic_stats_rules(self):
        """Add basic statistics rules."""
        rules = [
            Rule(
                id="unique_count",
                name="Unique Value Count",
                description="Count the number of unique values in a column",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_unique_count(series: pd.Series) -> Dict[str, Any]:
    unique_count = series.nunique()
    total_count = len(series)
    unique_ratio = unique_count / total_count if total_count > 0 else 0
    
    return {
        "unique_count": unique_count,
        "total_count": total_count,
        "unique_ratio": unique_ratio,
        "status": "passed" if unique_count > 0 else "warning",
        "message": f"Column has {unique_count} unique values ({unique_ratio:.2%} of total)"
    }
"""
            ),
            Rule(
                id="most_common_value",
                name="Most Common Value",
                description="Find the most common value and its frequency",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_most_common_value(series: pd.Series) -> Dict[str, Any]:
    value_counts = series.value_counts(dropna=False)
    
    if len(value_counts) == 0:
        return {
            "most_common_value": None,
            "frequency": 0,
            "frequency_ratio": 0,
            "status": "warning",
            "message": "No values found in column"
        }
    
    most_common = value_counts.index[0]
    frequency = value_counts.iloc[0]
    frequency_ratio = frequency / len(series)
    
    return {
        "most_common_value": most_common,
        "frequency": frequency,
        "frequency_ratio": frequency_ratio,
        "status": "passed",
        "message": f"Most common value: {most_common} (appears {frequency} times, {frequency_ratio:.2%})"
    }
"""
            ),
            Rule(
                id="null_analysis",
                name="Null Value Analysis",
                description="Analyze null values in the column",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_null_analysis(series: pd.Series) -> Dict[str, Any]:
    null_count = series.isnull().sum()
    total_count = len(series)
    null_ratio = null_count / total_count if total_count > 0 else 0
    
    if null_ratio == 0:
        status = "passed"
        message = "No null values found"
    elif null_ratio <= 0.05:
        status = "passed"
        message = f"Low null ratio: {null_ratio:.2%}"
    elif null_ratio <= 0.25:
        status = "warning"
        message = f"Medium null ratio: {null_ratio:.2%}"
    else:
        status = "failed"
        message = f"High null ratio: {null_ratio:.2%}"
    
    return {
        "null_count": null_count,
        "total_count": total_count,
        "null_ratio": null_ratio,
        "status": status,
        "message": message
    }
"""
            )
        ]
        
        for rule in rules:
            self.rules[rule.id] = rule
    
    def _add_numeric_rules(self):
        """Add numeric-specific rules."""
        rules = [
            Rule(
                id="numeric_stats",
                name="Numeric Statistics",
                description="Calculate basic numeric statistics",
                category="numeric_stats",
                column_types=["numeric"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_numeric_stats(series: pd.Series) -> Dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    stats = {
        "mean": numeric_series.mean(),
        "median": numeric_series.median(),
        "std": numeric_series.std(),
        "min": numeric_series.min(),
        "max": numeric_series.max(),
        "q25": numeric_series.quantile(0.25),
        "q75": numeric_series.quantile(0.75),
        "skewness": numeric_series.skew(),
        "kurtosis": numeric_series.kurtosis()
    }
    
    return {
        "statistics": stats,
        "status": "passed",
        "message": "Numeric statistics calculated successfully"
    }
"""
            ),
            Rule(
                id="outlier_detection",
                name="Outlier Detection",
                description="Detect outliers using IQR method",
                category="outlier_detection",
                column_types=["numeric"],
                diversity_levels=["medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_outlier_detection(series: pd.Series) -> Dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    
    if len(numeric_series) < 4:
        return {
            "outlier_count": 0,
            "outlier_ratio": 0,
            "outliers": [],
            "status": "warning",
            "message": "Insufficient data for outlier detection"
        }
    
    Q1 = numeric_series.quantile(0.25)
    Q3 = numeric_series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
    outlier_ratio = len(outliers) / len(numeric_series)
    
    status = "warning" if outlier_ratio > 0.1 else "passed"
    message = f"Found {len(outliers)} outliers ({outlier_ratio:.2%} of data)"
    
    return {
        "outlier_count": len(outliers),
        "outlier_ratio": outlier_ratio,
        "outliers": outliers.tolist()[:10],  # Limit to first 10 outliers
        "bounds": {"lower": lower_bound, "upper": upper_bound},
        "status": status,
        "message": message
    }
"""
            )
        ]
        
        for rule in rules:
            self.rules[rule.id] = rule
    
    def _add_textual_rules(self):
        """Add textual-specific rules."""
        rules = [
            Rule(
                id="length_analysis",
                name="String Length Analysis",
                description="Analyze string lengths in textual columns",
                category="length_analysis",
                column_types=["textual"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_length_analysis(series: pd.Series) -> Dict[str, Any]:
    string_series = series.astype(str)
    lengths = string_series.str.len()
    
    stats = {
        "mean_length": lengths.mean(),
        "median_length": lengths.median(),
        "min_length": lengths.min(),
        "max_length": lengths.max(),
        "std_length": lengths.std()
    }
    
    return {
        "length_statistics": stats,
        "status": "passed",
        "message": f"String lengths range from {stats['min_length']} to {stats['max_length']} characters"
    }
"""
            ),
            Rule(
                id="text_patterns",
                name="Text Pattern Analysis",
                description="Analyze common text patterns",
                category="text_patterns",
                column_types=["textual"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_text_patterns(series: pd.Series) -> Dict[str, Any]:
    string_series = series.astype(str)
    
    # Check for common patterns
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    
    email_matches = string_series.str.match(email_pattern, na=False).sum()
    phone_matches = string_series.str.match(phone_pattern, na=False).sum()
    url_matches = string_series.str.match(url_pattern, na=False).sum()
    
    total_non_null = len(string_series.dropna())
    
    patterns = {
        "email_like": {"count": email_matches, "ratio": email_matches / total_non_null if total_non_null > 0 else 0},
        "phone_like": {"count": phone_matches, "ratio": phone_matches / total_non_null if total_non_null > 0 else 0},
        "url_like": {"count": url_matches, "ratio": url_matches / total_non_null if total_non_null > 0 else 0}
    }
    
    return {
        "patterns": patterns,
        "status": "passed",
        "message": "Text pattern analysis completed"
    }
"""
            )
        ]
        
        for rule in rules:
            self.rules[rule.id] = rule
    
    def _add_temporal_rules(self):
        """Add temporal-specific rules."""
        rules = [
            Rule(
                id="date_validation",
                name="Date Format Validation",
                description="Validate date formats and ranges",
                category="date_validation",
                column_types=["temporal"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_date_validation(series: pd.Series) -> Dict[str, Any]:
    try:
        # Try to convert to datetime
        datetime_series = pd.to_datetime(series, errors='coerce')
        valid_count = datetime_series.notna().sum()
        total_count = len(series)
        valid_ratio = valid_count / total_count if total_count > 0 else 0
        
        if valid_ratio == 1.0:
            status = "passed"
            message = "All dates are valid"
        elif valid_ratio >= 0.95:
            status = "warning"
            message = f"Most dates are valid ({valid_ratio:.2%})"
        else:
            status = "failed"
            message = f"Many invalid dates ({valid_ratio:.2%} valid)"
        
        date_range = {
            "earliest": datetime_series.min(),
            "latest": datetime_series.max(),
            "span_days": (datetime_series.max() - datetime_series.min()).days if valid_count > 1 else 0
        }
        
        return {
            "valid_count": valid_count,
            "total_count": total_count,
            "valid_ratio": valid_ratio,
            "date_range": date_range,
            "status": status,
            "message": message
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Date validation failed: {str(e)}"
        }
"""
            )
        ]
        
        for rule in rules:
            self.rules[rule.id] = rule
    
    def _add_cross_column_rules(self):
        """Add cross-column rules."""
        rules = [
            Rule(
                id="identicality",
                name="Column Identicality",
                description="Check for identical columns",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template="""
def check_identicality(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    series1 = df[col1]
    series2 = df[col2]
    
    # Check if columns are identical
    are_identical = series1.equals(series2)
    
    # Check if columns are identical when ignoring nulls
    series1_no_nulls = series1.dropna()
    series2_no_nulls = series2.dropna()
    
    if len(series1_no_nulls) == len(series2_no_nulls):
        are_identical_no_nulls = series1_no_nulls.equals(series2_no_nulls)
    else:
        are_identical_no_nulls = False
    
    return {
        "are_identical": are_identical,
        "are_identical_no_nulls": are_identical_no_nulls,
        "status": "warning" if are_identical else "passed",
        "message": f"Columns {'are' if are_identical else 'are not'} identical"
    }
"""
            ),
            Rule(
                id="correlation",
                name="Numeric Correlation",
                description="Calculate correlation between numeric columns",
                category="cross_column",
                column_types=["numeric"],
                diversity_levels=["medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template="""
def check_correlation(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    try:
        # Convert to numeric, coercing errors to NaN
        series1 = pd.to_numeric(df[col1], errors='coerce')
        series2 = pd.to_numeric(df[col2], errors='coerce')
        
        # Calculate correlation
        correlation = series1.corr(series2)
        
        if pd.isna(correlation):
            return {
                "correlation": None,
                "strength": "unknown",
                "status": "warning",
                "message": "Could not calculate correlation (insufficient data or all NaN values)"
            }
        
        # Determine correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        return {
            "correlation": correlation,
            "strength": strength,
            "status": "passed",
            "message": f"Correlation: {correlation:.3f} ({strength})"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Correlation calculation failed: {str(e)}"
        }
"""
            )
        ]
        
        for rule in rules:
            self.rules[rule.id] = rule
    
    def get_relevant_rules(self, attributes: ColumnAttributes, enable_cross_column: bool = True) -> List[Rule]:
        """
        Get rules that are relevant for a column based on its attributes.
        
        Args:
            attributes: Column attributes
            enable_cross_column: Whether to include cross-column rules
            
        Returns:
            List of relevant rules
        """
        relevant_rules = []
        
        for rule in self.rules.values():
            # Check if rule applies to this column type
            if attributes.type_category.value not in rule.column_types:
                continue
            
            # Check if rule applies to this diversity level
            if attributes.diversity_level.value not in rule.diversity_levels:
                continue
            
            # Check if rule applies to this nullability level
            if attributes.nullability_level.value not in rule.nullability_levels:
                continue
            
            # Check cross-column requirement
            if rule.requires_cross_column and not enable_cross_column:
                continue
            
            relevant_rules.append(rule)
        
        return relevant_rules
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[Rule]:
        """List all available rules."""
        return list(self.rules.values())
    
    def get_rules_by_category(self, category: str) -> List[Rule]:
        """Get all rules in a specific category."""
        return [rule for rule in self.rules.values() if rule.category == category]
