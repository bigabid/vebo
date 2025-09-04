"""
Rule engine for managing and executing data profiling rules.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from functools import lru_cache
import hashlib
import warnings

# Suppress pandas datetime parsing warnings that occur during coercion
warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually, falling back to `dateutil`')

from .meta_rules import ColumnAttributes, MetaRuleDetector, TypeCategory, DiversityLevel


class RuleStatus(Enum):
    """Status of a rule execution indicating interest level."""
    PENDING = "pending"
    RUNNING = "running"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SKIPPED = "skipped"
    # Keep old values for backward compatibility
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"


class RulePriority(Enum):
    """Priority levels for rules (compat with extended tests)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RuleComplexity(Enum):
    """Complexity levels for rules (compat with extended tests)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RuleCategory(Enum):
    """Rule categories (optional semantic grouping)."""
    BASIC_STATS = "basic_stats"
    NUMERIC_STATS = "numeric_stats"
    TEXT_PATTERNS = "text_patterns"
    CROSS_COLUMN = "cross_column"
    DATE_VALIDATION = "date_validation"
    PRIVACY_SECURITY = "privacy_security"
    ADVANCED_CROSS_COLUMN = "advanced_cross_column"
    PERFORMANCE = "performance"
    OTHER = "other"


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

    def to_dict(self) -> Dict[str, Any]:
        """Compatibility helper used by some tests."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "status": self.status.value if isinstance(self.status, RuleStatus) else str(self.status),
            "score": self.score,
            "message": self.message,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
        }


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
    # Optional extended fields for compatibility with broader tests
    priority: RulePriority = RulePriority.MEDIUM
    complexity: RuleComplexity = RuleComplexity.MEDIUM
    enabled: bool = True
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}

    def is_applicable(self, attributes: ColumnAttributes) -> bool:
        """Compatibility helper: determine if the rule applies to the given attributes."""
        if attributes.type_category.value not in self.column_types:
            return False
        if attributes.diversity_level.value not in self.diversity_levels:
            return False
        if attributes.nullability_level.value not in self.nullability_levels:
            return False
        return True


class RuleEngine:
    """
    Engine for managing and executing data profiling rules.
    """
    
    def __init__(self, load_builtin: bool = True):
        """Initialize the rule engine."""
        self.rules: Dict[str, Rule] = {}
        self.meta_detector = MetaRuleDetector()
        # Compatibility maps for richer test suite expectations
        self.rule_categories: Dict[str, List[str]] = {}
        self.rule_dependencies: Dict[str, List[str]] = {}
        self.rule_priorities: Dict[str, RulePriority] = {}
        self.rule_complexities: Dict[str, RuleComplexity] = {}
        self._user_rule_ids: set[str] = set()
        if load_builtin:
            self._load_builtin_rules()
    
    def _load_builtin_rules(self):
        """Load built-in rules based on the rule categories."""
        self._add_basic_stats_rules()
        self._add_numeric_rules()
        self._add_textual_rules()
        self._add_array_rules()
        self._add_dictionary_rules()
        self._add_temporal_rules()
        self._add_cross_column_rules()
        self._add_privacy_rules()
        self._add_advanced_cross_column_rules()
        self._add_performance_rules()
    
    def _add_basic_stats_rules(self):
        """Add basic statistics rules."""
        rules = [
            Rule(
                id="unique_count",
                name="Unique Value Count",
                description="Count the number of unique values in a column",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_unique_count(series: pd.Series) -> Dict[str, Any]:
    # Use len(set()) for better performance, excluding NaN values like nunique()
    unique_count = len(set(series.dropna()))
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
                column_types=["numeric", "textual", "temporal", "boolean", "categorical", "array", "dictionary"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_most_common_value(series: pd.Series) -> Dict[str, Any]:
    # Check if unique count is 1 (constant column) - skip if so
    unique_count = len(set(series.dropna()))
    if unique_count <= 1:
        # For constant columns, most common is obvious and redundant
        non_null_values = series.dropna()
        if len(non_null_values) > 0:
            constant_value = non_null_values.iloc[0]
            return {
                "most_common_value": constant_value,
                "frequency": len(non_null_values),
                "frequency_ratio": len(non_null_values) / len(series),
                "is_constant_column": True,
                "status": "passed",
                "message": f"Constant column with value: {constant_value}"
            }
        else:
            return {
                "most_common_value": None,
                "frequency": 0,
                "frequency_ratio": 0,
                "is_constant_column": True,
                "status": "warning",
                "message": "All values are null"
            }
    
    value_counts_with_null = series.value_counts(dropna=False)
    value_counts_no_null = series.value_counts(dropna=True)
    
    if len(value_counts_with_null) == 0:
        return {
            "most_common_value": None,
            "frequency": 0,
            "frequency_ratio": 0,
            "status": "warning",
            "message": "No values found in column"
        }
    
    # Get most common value (including null)
    most_common = value_counts_with_null.index[0]
    frequency = value_counts_with_null.iloc[0]
    frequency_ratio = frequency / len(series)
    
    result = {
        "most_common_value": most_common,
        "frequency": frequency,
        "frequency_ratio": frequency_ratio,
        "status": "passed"
    }
    
    # If most common value is null, also provide most common non-null value
    if pd.isna(most_common) and len(value_counts_no_null) > 0:
        most_common_non_null = value_counts_no_null.index[0]
        frequency_non_null = value_counts_no_null.iloc[0]
        frequency_ratio_non_null = frequency_non_null / len(series.dropna())
        
        result["most_common_non_null_value"] = most_common_non_null
        result["most_common_non_null_frequency"] = frequency_non_null
        result["most_common_non_null_frequency_ratio"] = frequency_ratio_non_null
        result["message"] = f"Most common: null ({frequency_ratio:.2%}), most common non-null: {most_common_non_null} ({frequency_ratio_non_null:.2%})"
    else:
        result["message"] = f"Most common value: {most_common} (appears {frequency} times, {frequency_ratio:.2%})"
    
    return result
"""
            ),
            Rule(
                id="null_analysis",
                name="Null Value Analysis",
                description="Analyze null values in the column",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
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
            ),
            Rule(
                id="top_k_frequencies",
                name="Top-K Frequencies",
                description="Top-k most common values, frequencies, and 'other' proportion (excluding nulls)",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical", "array", "dictionary"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_top_k_frequencies(series: pd.Series) -> Dict[str, Any]:
    k = 10
    # Exclude nulls from top values - they're not meaningful for "top values"
    value_counts = series.value_counts(dropna=True)
    non_null_count = len(series.dropna())
    
    if len(value_counts) == 0:
        return {
            "top_k": [],
            "other_count": 0,
            "other_ratio": 0,
            "status": "warning",
            "message": "No non-null values found for top-k frequencies"
        }
    
    top_k = value_counts.head(k)
    other_count = max(non_null_count - int(top_k.sum()), 0)
    other_ratio = other_count / non_null_count if non_null_count > 0 else 0
    
    top_values = [
        {"value": v, "count": int(c), "ratio": (int(c) / non_null_count if non_null_count > 0 else 0)}
        for v, c in zip(top_k.index.tolist(), top_k.values.tolist())
    ]
    
    return {
        "top_k": top_values,
        "other_count": other_count,
        "other_ratio": other_ratio,
        "non_null_count": non_null_count,
        "status": "passed",
        "message": f"Computed top-{k} frequencies (excluding nulls)"
    }
"""
            ),
            Rule(
                id="duplicate_value_analysis",
                name="Duplicate Value Analysis",
                description="Counts and proportion of duplicate values (beyond first occurrences)",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_duplicate_value_analysis(series: pd.Series) -> Dict[str, Any]:
    total_count = len(series)
    unique_count = series.nunique(dropna=False)
    duplicate_count = max(total_count - unique_count, 0)
    duplicate_ratio = duplicate_count / total_count if total_count > 0 else 0
    return {
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_ratio,
        "unique_count": unique_count,
        "status": "passed",
        "message": f"{duplicate_count} duplicates ({duplicate_ratio:.2%})"
    }
"""
            ),
            Rule(
                id="parseability_analysis",
                name="Parseability Analysis",
                description="Percentage of values parseable as integer, float, datetime, JSON",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_parseability_analysis(series: pd.Series) -> Dict[str, Any]:
    import json as _json
    import warnings
    s = series.dropna()
    total = len(s)
    if total == 0:
        return {"status": "warning", "message": "No non-null values", "parseable_int_ratio": 0, "parseable_float_ratio": 0, "parseable_datetime_ratio": 0, "parseable_json_ratio": 0}
    as_numeric = pd.to_numeric(s, errors='coerce')
    int_mask = as_numeric.notna() & (as_numeric % 1 == 0)
    float_mask = as_numeric.notna()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually, falling back to `dateutil`')
        as_dt = pd.to_datetime(s, errors='coerce', utc=True)
    dt_mask = as_dt.notna()
    def _is_json_str(x):
        if not isinstance(x, str):
            return False
        try:
            _json.loads(x)
            return True
        except Exception:
            return False
    json_mask = s.apply(_is_json_str)
    return {
        "parseable_int_ratio": float(int_mask.sum()) / total,
        "parseable_float_ratio": float(float_mask.sum()) / total,
        "parseable_datetime_ratio": float(dt_mask.sum()) / total,
        "parseable_json_ratio": float(json_mask.sum()) / total,
        "status": "passed",
        "message": "Parseability computed"
    }
"""
            ),
            Rule(
                id="stability_entropy",
                name="Value Stability and Entropy",
                description="Share of repeated values and entropy-based diversity",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_stability_entropy(series: pd.Series) -> Dict[str, Any]:
    non_null = series.dropna()
    total = len(non_null)
    if total == 0:
        return {"status": "warning", "message": "No non-null values", "repeated_share": 0, "entropy": 0, "normalized_entropy": 0}
    vc = non_null.value_counts()
    repeated_occurrences = int(vc[vc > 1].sum())
    repeated_share = float(repeated_occurrences) / total if total > 0 else 0.0
    probs = (vc / total).values.astype(float)
    entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
    max_entropy = float(np.log2(len(vc))) if len(vc) > 0 else 1.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return {
        "repeated_share": repeated_share,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "status": "passed",
        "message": "Computed stability and entropy"
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_numeric_rules(self):
        """Add numeric-specific rules."""
        rules = [
            Rule(
                id="numeric_stats",
                name="Numeric Statistics",
                description="Calculate basic numeric statistics",
                category="numeric_stats",
                column_types=["numeric"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_numeric_stats(series: pd.Series) -> Dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors='coerce')
    non_null_numeric = numeric_series.dropna()
    
    # Check if this is a constant column (unique count <= 1)
    unique_count = len(set(non_null_numeric)) if len(non_null_numeric) > 0 else 0
    
    if unique_count <= 1:
        # For constant columns, most stats are redundant
        if len(non_null_numeric) > 0:
            constant_value = float(non_null_numeric.iloc[0])
            stats = {
                "mean": constant_value,
                "median": constant_value,
                "min": constant_value,
                "max": constant_value,
                "std": 0.0,
                "q25": constant_value,
                "q75": constant_value,
                "skewness": 0.0,
                "kurtosis": 0.0,
                "range": 0.0
            }
            return {
                "statistics": stats,
                "is_constant_column": True,
                "status": "passed",
                "message": f"Constant numeric column with value: {constant_value}"
            }
        else:
            return {
                "statistics": {},
                "is_constant_column": True,
                "status": "warning",
                "message": "All numeric values are null"
            }
    
    # For non-constant columns, calculate full statistics
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
    
    # Add range (max - min)
    if not pd.isna(stats["max"]) and not pd.isna(stats["min"]):
        stats["range"] = stats["max"] - stats["min"]
    
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
                code_template=r"""
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
            ),
            Rule(
                id="outlier_detection_zscore",
                name="Outlier Detection (Z-Score)",
                description="Detect outliers using Z-score method",
                category="outlier_detection",
                column_types=["numeric"],
                diversity_levels=["medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_outlier_detection_zscore(series: pd.Series) -> Dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric_series) < 4:
        return {"outlier_count": 0, "outlier_ratio": 0, "outliers": [], "status": "warning", "message": "Insufficient data"}
    mean = numeric_series.mean()
    std = numeric_series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return {"outlier_count": 0, "outlier_ratio": 0, "outliers": [], "status": "warning", "message": "Zero variance"}
    z = (numeric_series - mean) / std
    threshold = 3.0
    mask = z.abs() > threshold
    outliers = numeric_series[mask]
    ratio = len(outliers) / len(numeric_series)
    return {
        "outlier_count": int(len(outliers)),
        "outlier_ratio": float(ratio),
        "z_threshold": threshold,
        "status": "warning" if ratio > 0.1 else "passed",
        "message": f"Found {int(len(outliers))} z-score outliers ({ratio:.2%})"
    }
"""
            ),
            Rule(
                id="numeric_histogram_quantiles",
                name="Histogram and Quantiles",
                description="Histogram bins and key percentiles",
                category="numeric_stats",
                column_types=["numeric"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_numeric_histogram_quantiles(series: pd.Series) -> Dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric_series) == 0:
        return {"status": "warning", "message": "No numeric values", "histogram": {}, "quantiles": {}}
    counts, bin_edges = np.histogram(numeric_series, bins=min(20, max(5, int(np.sqrt(len(numeric_series))))) )
    quantile_points = [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]
    q_values = {str(q): float(numeric_series.quantile(q)) for q in quantile_points}
    return {
        "histogram": {"counts": counts.tolist(), "bin_edges": [float(x) for x in bin_edges.tolist()]},
        "quantiles": q_values,
        "status": "passed",
        "message": "Histogram and quantiles computed"
    }
"""
            ),
            Rule(
                id="modality_estimation",
                name="Modality Estimation",
                description="Estimate distribution modality from histogram peaks",
                category="numeric_stats",
                column_types=["numeric"],
                diversity_levels=["medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_modality_estimation(series: pd.Series) -> Dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric_series) < 10:
        return {"status": "warning", "message": "Insufficient data for modality", "modality": "unknown"}
    counts, _ = np.histogram(numeric_series, bins=20)
    if len(counts) >= 3:
        smooth = np.convolve(counts, np.ones(3)/3, mode='same')
    else:
        smooth = counts
    threshold = max(smooth) * 0.1 if max(smooth) > 0 else 0
    peaks = 0
    for i in range(1, len(smooth)-1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] >= threshold:
            peaks += 1
    if peaks <= 1:
        modality = "unimodal"
    elif peaks == 2:
        modality = "bimodal"
    else:
        modality = "multimodal"
    return {"modality": modality, "num_peaks": int(peaks), "status": "passed", "message": f"{modality} distribution"}
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_textual_rules(self):
        """Add textual-specific rules."""
        rules = [
            Rule(
                id="length_analysis",
                name="String Length Analysis",
                description="Analyze string lengths in textual columns",
                category="length_analysis",
                column_types=["textual"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
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
                id="whitespace_encoding_checks",
                name="Whitespace and Encoding Checks",
                description="Leading/trailing spaces, whitespace-only, non-printable characters",
                category="text_quality",
                column_types=["textual"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_whitespace_encoding_checks(series: pd.Series) -> Dict[str, Any]:
    s = series.astype(str)
    total = len(s)
    if total == 0:
        return {"status": "warning", "message": "Empty series"}
    leading_ws = s.str.match(r'^\s+').sum()
    trailing_ws = s.str.match(r'.*\s+$').sum()
    whitespace_only = s.str.match(r'^\s*$').sum()
    def _has_nonprintable(x: str) -> bool:
        try:
            return any(ord(ch) < 32 or ord(ch) == 127 for ch in x)
        except Exception:
            return False
    non_printable = s.apply(_has_nonprintable).sum()
    return {
        "leading_whitespace_ratio": float(leading_ws) / total,
        "trailing_whitespace_ratio": float(trailing_ws) / total,
        "whitespace_only_ratio": float(whitespace_only) / total,
        "non_printable_ratio": float(non_printable) / total,
        "status": "passed",
        "message": "Whitespace and encoding checks computed"
    }
"""
            ),
            Rule(
                id="text_patterns",
                name="Text Pattern Analysis",
                description="Analyze common text patterns",
                category="text_patterns",
                column_types=["textual"],
                diversity_levels=["medium", "high", "distinctive", "fully_unique"],  # Skip low diversity
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_text_patterns(series: pd.Series) -> Dict[str, Any]:
    # TextRegexInference is made available in the execution namespace
    # by the check executor
    regex_engine = TextRegexInference()
    
    string_series = series.astype(str)
    
    # Check for the basic patterns first
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
    url_pattern = r'^https?://[^\s/$\.?#]\.[^\s]*$'
    
    total_non_null = len(string_series.dropna())
    
    try:
        email_matches = string_series.str.match(email_pattern, na=False).sum()
        phone_matches = string_series.str.match(phone_pattern, na=False).sum()
        url_matches = string_series.str.match(url_pattern, na=False).sum()
    except Exception:
        email_matches = phone_matches = url_matches = 0
    
    basic_patterns = {
        "email_like": {"count": email_matches, "ratio": email_matches / total_non_null if total_non_null > 0 else 0},
        "phone_like": {"count": phone_matches, "ratio": phone_matches / total_non_null if total_non_null > 0 else 0},
        "url_like": {"count": url_matches, "ratio": url_matches / total_non_null if total_non_null > 0 else 0}
    }
    
    # Check if any known pattern matches most of the data (>80%)
    # If so, skip expensive regex inference
    skip_regex_inference = False
    max_known_ratio = max(basic_patterns[pattern]["ratio"] for pattern in basic_patterns)
    if max_known_ratio > 0.8:
        skip_regex_inference = True
        skip_reason = "Known pattern detected with high confidence"
    
    inferred_pattern_data = []
    if not skip_regex_inference:
        # Use the regex inference system only if no known patterns dominate
        inferred_patterns = regex_engine.infer_patterns(string_series, max_patterns=3)
        for pattern in inferred_patterns:
            inferred_pattern_data.append({
                "regex": pattern.pattern,
                "description": pattern.description,
                "match_count": pattern.match_count,
                "match_ratio": pattern.match_ratio,
                "confidence": pattern.confidence,
                "examples": pattern.examples
            })
    
    message = f"Text pattern analysis completed. Found {len(inferred_pattern_data)} inferred patterns."
    if skip_regex_inference:
        message += f" Skipped regex inference: {skip_reason}."
    
    return {
        "basic_patterns": basic_patterns,
        "inferred_patterns": inferred_pattern_data,
        "status": "passed",
        "message": message
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_array_rules(self):
        """Add array-specific rules."""
        rules = [
            Rule(
                id="array_length_analysis",
                name="Array Length Analysis",
                description="Analyze array lengths and statistics",
                category="array_analysis",
                column_types=["array"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_array_length_analysis(series: pd.Series) -> Dict[str, Any]:
    import json
    lengths = []
    invalid_count = 0
    
    for value in series.dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, list):
                lengths.append(len(parsed))
            else:
                invalid_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1
    
    if not lengths:
        return {
            "status": "warning",
            "message": "No valid arrays found"
        }
    
    length_stats = {
        "mean_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "total_arrays": len(lengths),
        "invalid_arrays": invalid_count
    }
    
    return {
        "array_length_statistics": length_stats,
        "status": "passed",
        "message": f"Arrays have lengths ranging from {length_stats['min_length']} to {length_stats['max_length']}"
    }
"""
            ),
            Rule(
                id="array_depth_analysis",
                name="Array Depth Analysis",
                description="Analyze nesting depth of arrays",
                category="depth_analysis",
                column_types=["array"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_array_depth_analysis(series: pd.Series) -> Dict[str, Any]:
    import json
    
    def get_max_depth(obj):
        if not isinstance(obj, (list, dict)):
            return 0
        if isinstance(obj, list) and not obj:
            return 1
        return 1 + max(get_max_depth(item) for item in obj) if obj else 1
    
    depths = []
    invalid_count = 0
    
    for value in series.dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, list):
                depth = get_max_depth(parsed)
                depths.append(depth)
            else:
                invalid_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1
    
    if not depths:
        return {
            "status": "warning",
            "message": "No valid arrays found"
        }
    
    depth_stats = {
        "mean_depth": sum(depths) / len(depths),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "total_arrays": len(depths),
        "invalid_arrays": invalid_count
    }
    
    return {
        "array_depth_statistics": depth_stats,
        "status": "passed",
        "message": f"Arrays have depths ranging from {depth_stats['min_depth']} to {depth_stats['max_depth']}"
    }
"""
            ),
            Rule(
                id="array_element_type_analysis",
                name="Array Element Type Analysis",
                description="Analyze types of elements within arrays",
                category="element_type_analysis",
                column_types=["array"],
                diversity_levels=["low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_array_element_type_analysis(series: pd.Series) -> Dict[str, Any]:
    import json
    from collections import defaultdict
    
    element_types = defaultdict(int)
    total_elements = 0
    invalid_count = 0
    
    for value in series.dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, list):
                for element in parsed:
                    total_elements += 1
                    element_type = type(element).__name__
                    element_types[element_type] += 1
            else:
                invalid_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1
    
    if not element_types:
        return {
            "status": "warning",
            "message": "No valid array elements found"
        }
    
    type_distribution = {
        elem_type: count / total_elements 
        for elem_type, count in element_types.items()
    }
    
    return {
        "element_type_distribution": type_distribution,
        "total_elements": total_elements,
        "unique_types": len(element_types),
        "most_common_type": max(element_types.items(), key=lambda x: x[1])[0],
        "status": "passed",
        "message": f"Arrays contain {len(element_types)} different element types across {total_elements} total elements"
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_dictionary_rules(self):
        """Add dictionary-specific rules."""
        rules = [
            Rule(
                id="dictionary_key_analysis",
                name="Dictionary Key Analysis",
                description="Analyze dictionary keys and their patterns",
                category="key_analysis",
                column_types=["dictionary"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_dictionary_key_analysis(series: pd.Series) -> Dict[str, Any]:
    import json
    from collections import defaultdict
    
    all_keys = set()
    key_counts = defaultdict(int)
    dict_count = 0
    invalid_count = 0
    
    for value in series.dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, dict):
                dict_count += 1
                for key in parsed.keys():
                    all_keys.add(key)
                    key_counts[key] += 1
            else:
                invalid_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1
    
    if not all_keys:
        return {
            "status": "warning",
            "message": "No valid dictionaries found"
        }
    
    # Calculate key consistency
    consistent_keys = [key for key, count in key_counts.items() if count == dict_count]
    
    key_stats = {
        "total_unique_keys": len(all_keys),
        "consistent_keys": len(consistent_keys),
        "most_common_keys": sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "total_dictionaries": dict_count,
        "invalid_dictionaries": invalid_count
    }
    
    return {
        "dictionary_key_statistics": key_stats,
        "status": "passed",
        "message": f"Dictionaries have {len(all_keys)} unique keys, {len(consistent_keys)} appear in all dictionaries"
    }
"""
            ),
            Rule(
                id="dictionary_depth_analysis",
                name="Dictionary Depth Analysis",
                description="Analyze nesting depth of dictionaries",
                category="depth_analysis",
                column_types=["dictionary"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_dictionary_depth_analysis(series: pd.Series) -> Dict[str, Any]:
    import json
    
    def get_max_depth(obj):
        if not isinstance(obj, (list, dict)):
            return 0
        if isinstance(obj, dict) and not obj:
            return 1
        if isinstance(obj, dict):
            return 1 + max(get_max_depth(value) for value in obj.values()) if obj else 1
        if isinstance(obj, list):
            return 1 + max(get_max_depth(item) for item in obj) if obj else 1
        return 0
    
    depths = []
    invalid_count = 0
    
    for value in series.dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, dict):
                depth = get_max_depth(parsed)
                depths.append(depth)
            else:
                invalid_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1
    
    if not depths:
        return {
            "status": "warning",
            "message": "No valid dictionaries found"
        }
    
    depth_stats = {
        "mean_depth": sum(depths) / len(depths),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "total_dictionaries": len(depths),
        "invalid_dictionaries": invalid_count
    }
    
    return {
        "dictionary_depth_statistics": depth_stats,
        "status": "passed",
        "message": f"Dictionaries have depths ranging from {depth_stats['min_depth']} to {depth_stats['max_depth']}"
    }
"""
            ),
            Rule(
                id="dictionary_schema_analysis",
                name="Dictionary Schema Analysis",
                description="Analyze schema consistency across dictionaries",
                category="schema_analysis",
                column_types=["dictionary"],
                diversity_levels=["low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_dictionary_schema_analysis(series: pd.Series) -> Dict[str, Any]:
    import json
    from collections import defaultdict
    
    schemas = defaultdict(int)  # Track different key combinations
    value_types = defaultdict(lambda: defaultdict(int))  # Track value types for each key
    dict_count = 0
    invalid_count = 0
    
    for value in series.dropna():
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, dict):
                dict_count += 1
                # Create schema signature (sorted keys)
                schema = tuple(sorted(parsed.keys()))
                schemas[schema] += 1
                
                # Track value types for each key
                for key, val in parsed.items():
                    val_type = type(val).__name__
                    value_types[key][val_type] += 1
            else:
                invalid_count += 1
        except (json.JSONDecodeError, TypeError):
            invalid_count += 1
    
    if not schemas:
        return {
            "status": "warning",
            "message": "No valid dictionaries found"
        }
    
    # Find most common schema
    most_common_schema = max(schemas.items(), key=lambda x: x[1])
    schema_consistency = most_common_schema[1] / dict_count if dict_count > 0 else 0
    
    schema_stats = {
        "unique_schemas": len(schemas),
        "most_common_schema": most_common_schema[0],
        "schema_consistency_ratio": schema_consistency,
        "total_dictionaries": dict_count,
        "value_type_analysis": dict(value_types),
        "invalid_dictionaries": invalid_count
    }
    
    return {
        "dictionary_schema_statistics": schema_stats,
        "status": "passed",
        "message": f"Found {len(schemas)} unique schemas, {schema_consistency:.1%} of dictionaries follow the most common schema"
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_temporal_rules(self):
        """Add temporal-specific rules."""
        rules = [
            Rule(
                id="date_validation",
                name="Date Format Validation",
                description="Validate date formats and ranges",
                category="date_validation",
                column_types=["temporal"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_date_validation(series: pd.Series) -> Dict[str, Any]:
    import warnings
    try:
        # Try to convert to datetime
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Could not infer format, so each element will be parsed individually, falling back to `dateutil`')
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
            self._register_rule(rule, builtin=True)
    
    def _add_cross_column_rules(self):
        """Add cross-column rules."""
        rules = [
            Rule(
                id="identicality",
                name="Column Identicality",
                description="Check for identical columns",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template=r"""
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
        "status": "high" if are_identical else "low",
        "message": f"Columns {'are' if are_identical else 'are not'} identical",
        "should_skip_other_rules": are_identical  # Signal to skip other rules for this pair
    }
"""
            ),
            Rule(
                id="correlation",
                name="Numeric Correlation",
                description="Calculate correlation between numeric columns",
                category="cross_column",
                column_types=["numeric"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template=r"""
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
                "status": "skipped",
                "message": "Could not calculate correlation (insufficient data or all NaN values)"
            }
        
        # Determine correlation strength and interest level
        abs_corr = abs(correlation)
        if abs_corr >= 0.8:
            strength = "strong"
            status = "high"
        elif abs_corr >= 0.5:
            strength = "moderate"
            status = "medium"
        else:
            strength = "weak" if abs_corr >= 0.3 else "very weak"
            status = "low"
        
        return {
            "correlation": correlation,
            "strength": strength,
            "status": status,
            "message": f"Correlation: {correlation:.3f} ({strength})"
        }
    except Exception as e:
        return {
            "status": "high",
            "message": f"Correlation calculation failed: {str(e)}"
        }
"""
            ),
            Rule(
                id="missingness_relationships",
                name="Missingness Relationships",
                description="When nulls in one column coincide with nulls in another",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template=r"""
def check_missingness_relationships(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    s1 = df[col1]
    s2 = df[col2]
    n = len(df)
    both_null = (s1.isna() & s2.isna()).sum()
    s1_null_only = (s1.isna() & s2.notna()).sum()
    s2_null_only = (s2.isna() & s1.notna()).sum()
    co_null_ratio = both_null / n if n > 0 else 0
    
    # Determine interest level based on synchronized missingness
    # High interest when columns have strong synchronized missingness pattern
    status = "high" if co_null_ratio >= 0.3 else "low"
    
    return {
        "both_null_count": int(both_null),
        "s1_null_only_count": int(s1_null_only),
        "s2_null_only_count": int(s2_null_only),
        "co_null_ratio": float(co_null_ratio),
        "status": status,
        "message": f"Synchronized missingness: {co_null_ratio:.1%} of rows"
    }
"""
            ),
            Rule(
                id="functional_dependency",
                name="Functional Dependency Approximation",
                description="Approximate functional dependency in both directions and return the strongest",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template=r"""
def check_functional_dependency(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    import pandas as pd
    import numpy as np
    import sys
    import os
    
    # Import statistical enhancements
    try:
        from statistical_enhancements import StatisticalEnhancements
    except ImportError:
        # Fallback if statistical_enhancements not available - use basic implementation
        return check_functional_dependency_basic(df, col1, col2)
    
    enhancer = StatisticalEnhancements()
    
    # LAYER 3: Check statistical power first (data-size adaptive)
    power_check = enhancer.should_proceed_with_analysis(df, col1, col2, "fd")
    if not power_check['should_proceed']:
        return {
            "status": "skipped",
            "message": f"Skipped: {power_check['reason']}",
            "fd_holds_ratio": 0,
            "reason": power_check['reason'],
            "skip_details": power_check['power_details'],
            "enhanced_analysis": True
        }
    
    # Original categorical checks (keep existing logic for compatibility)
    def is_categorical_like(series):
        if pd.api.types.is_categorical_dtype(series):
            return True
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            return unique_ratio < 0.7
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            return unique_ratio < 0.3 and series.nunique() <= 50
        return False
    
    def is_fully_unique(series):
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        return unique_ratio >= 0.95
    
    col1_series = df[col1].dropna()
    col2_series = df[col2].dropna()
    
    # Check original skip conditions for backward compatibility
    col1_is_categorical = is_categorical_like(col1_series)
    col2_is_categorical = is_categorical_like(col2_series)
    col1_is_unique = is_fully_unique(col1_series)
    col2_is_unique = is_fully_unique(col2_series)
    
    col1_unique_ratio = col1_series.nunique() / len(col1_series) if len(col1_series) > 0 else 0
    col2_unique_ratio = col2_series.nunique() / len(col2_series) if len(col2_series) > 0 else 0
    
    skip_reasons = []
    
    if not col1_is_categorical or not col2_is_categorical:
        non_cat_cols = []
        if not col1_is_categorical:
            non_cat_cols.append(f"{col1} ({col1_series.dtype})")
        if not col2_is_categorical:
            non_cat_cols.append(f"{col2} ({col2_series.dtype})")
        skip_reasons.append(f"Non-categorical columns: {', '.join(non_cat_cols)}")
    
    if col1_is_unique or col2_is_unique:
        unique_cols = []
        if col1_is_unique:
            unique_cols.append(f"{col1} ({col1_unique_ratio:.1%})")
        if col2_is_unique:
            unique_cols.append(f"{col2} ({col2_unique_ratio:.1%})")
        skip_reasons.append(f"Fully unique columns: {', '.join(unique_cols)}")
    
    if skip_reasons:
        return {
            "status": "skipped", 
            "message": f"Skipped: {'; '.join(skip_reasons)}",
            "fd_holds_ratio": 0,
            "reason": "skip_conditions_met",
            "skip_details": {
                "non_categorical": not col1_is_categorical or not col2_is_categorical,
                "fully_unique": col1_is_unique or col2_is_unique,
                "col1_categorical": col1_is_categorical,
                "col2_categorical": col2_is_categorical,
                "col1_unique_ratio": col1_unique_ratio,
                "col2_unique_ratio": col2_unique_ratio
            }
        }
    
    # LAYER 4: Enhanced analysis with confidence measures
    return enhancer.enhanced_functional_dependency_with_confidence(df, col1, col2)

def check_functional_dependency_basic(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    # Fallback basic implementation if enhanced version unavailable
    import pandas as pd
    
    def check_direction(df, source_col, target_col):
        grouped_data = df[[source_col, target_col]].dropna().groupby(source_col, dropna=False)[target_col]
        g = grouped_data.apply(lambda x: len(set(x)))
        if len(g) == 0:
            return {"fd_holds_ratio": 0, "violating_groups": 0, "total_groups": 0}
        violating = (g > 1).sum()
        fd_holds_ratio = 1.0 - (violating / len(g))
        return {
            "fd_holds_ratio": float(fd_holds_ratio),
            "violating_groups": int(violating),
            "total_groups": int(len(g))
        }
    
    fd_col1_to_col2 = check_direction(df, col1, col2)
    fd_col2_to_col1 = check_direction(df, col2, col1)
    
    if fd_col1_to_col2["fd_holds_ratio"] >= fd_col2_to_col1["fd_holds_ratio"]:
        best_direction = f"{col1} -> {col2}"
        best_ratio = fd_col1_to_col2["fd_holds_ratio"]
        best_violating = fd_col1_to_col2["violating_groups"]
        best_total = fd_col1_to_col2["total_groups"]
    else:
        best_direction = f"{col2} -> {col1}"
        best_ratio = fd_col2_to_col1["fd_holds_ratio"]
        best_violating = fd_col2_to_col1["violating_groups"]
        best_total = fd_col2_to_col1["total_groups"]
    
    if best_total == 0:
        return {"status": "low", "message": "Insufficient data", "fd_holds_ratio": 0}
    
    status = "high" if best_ratio >= 0.95 else ("medium" if best_ratio >= 0.8 else "low")
    
    return {
        "best_direction": best_direction,
        "fd_holds_ratio": float(best_ratio),
        "violating_groups": int(best_violating),
        "total_groups": int(best_total),
        "col1_to_col2_ratio": float(fd_col1_to_col2["fd_holds_ratio"]),
        "col2_to_col1_ratio": float(fd_col2_to_col1["fd_holds_ratio"]),
        "status": status,
        "message": f"Best FD: {best_direction} holds in {best_ratio:.2%} of groups"
    }
"""
            ),
            Rule(
                id="composite_uniqueness",
                name="Composite Uniqueness",
                description="Whether a combination of two columns yields unique rows",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template=r"""
def check_composite_uniqueness(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    # Check if either column is 100% unique by itself
    total = len(df)
    
    # Check individual column uniqueness
    col1_unique_count = df[col1].nunique(dropna=True)
    col2_unique_count = df[col2].nunique(dropna=True)
    col1_non_null_count = len(df[col1].dropna())
    col2_non_null_count = len(df[col2].dropna())
    
    col1_unique_ratio = col1_unique_count / col1_non_null_count if col1_non_null_count > 0 else 0
    col2_unique_ratio = col2_unique_count / col2_non_null_count if col2_non_null_count > 0 else 0
    
    # Skip composite check if either column is 100% unique
    if col1_unique_ratio == 1.0:
        return {
            "duplicate_pairs": 0,
            "unique_pair_ratio": 1.0,
            "status": "skipped",
            "message": f"Skipped: {col1} is already 100% unique",
            "skipped_reason": f"Column '{col1}' has 100% uniqueness",
            "individual_uniqueness": {
                "col1_unique_ratio": float(col1_unique_ratio),
                "col2_unique_ratio": float(col2_unique_ratio)
            }
        }
    
    if col2_unique_ratio == 1.0:
        return {
            "duplicate_pairs": 0,
            "unique_pair_ratio": 1.0,
            "status": "skipped",
            "message": f"Skipped: {col2} is already 100% unique",
            "skipped_reason": f"Column '{col2}' has 100% uniqueness",
            "individual_uniqueness": {
                "col1_unique_ratio": float(col1_unique_ratio),
                "col2_unique_ratio": float(col2_unique_ratio)
            }
        }
    
    # Proceed with normal composite uniqueness check
    dup_count = df.duplicated(subset=[col1, col2]).sum()
    unique_ratio = 1.0 - (dup_count / total if total > 0 else 0)
    
    # Determine interest level based on uniqueness
    if unique_ratio == 1.0:
        status = "high"  # 100% unique pairs - very interesting
    elif unique_ratio >= 0.99:
        status = "medium"  # Nearly unique pairs - moderately interesting
    else:
        status = "low"  # Many duplicates - low interest
    
    return {
        "duplicate_pairs": int(dup_count),
        "unique_pair_ratio": float(unique_ratio),
        "status": status,
        "message": f"Unique pair ratio: {unique_ratio:.2%}",
        "individual_uniqueness": {
            "col1_unique_ratio": float(col1_unique_ratio),
            "col2_unique_ratio": float(col2_unique_ratio)
        }
    }
"""
            ),

            Rule(
                id="categorical_association_cramers_v",
                name="Categorical Association (Cramér's V)",
                description="Association strength between two categorical columns",
                category="cross_column",
                column_types=["categorical", "textual"],
                diversity_levels=["binary", "low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template=r"""
def check_categorical_association_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    import pandas as pd
    import numpy as np
    import sys
    import os
    
    # Import statistical enhancements
    try:
        from statistical_enhancements import StatisticalEnhancements
    except ImportError:
        # Fallback if statistical_enhancements not available - use basic implementation
        return check_categorical_association_cramers_v_basic(df, col1, col2)
    
    enhancer = StatisticalEnhancements()
    
    # LAYER 3: Check statistical power first (data-size adaptive)
    power_check = enhancer.should_proceed_with_analysis(df, col1, col2, "cramers_v")
    if not power_check['should_proceed']:
        return {
            "status": "skipped",
            "message": f"Skipped: {power_check['reason']}",
            "cramers_v": None,
            "reason": power_check['reason'],
            "skip_details": power_check['power_details'],
            "enhanced_analysis": True
        }
    
    # Original uniqueness checks for backward compatibility
    def is_fully_unique(series):
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        return unique_ratio >= 0.95
    
    col1_series = df[col1].dropna()
    col2_series = df[col2].dropna()
    
    col1_is_unique = is_fully_unique(col1_series)
    col2_is_unique = is_fully_unique(col2_series)
    
    if col1_is_unique or col2_is_unique:
        col1_unique_ratio = col1_series.nunique() / len(col1_series) if len(col1_series) > 0 else 0
        col2_unique_ratio = col2_series.nunique() / len(col2_series) if len(col2_series) > 0 else 0
        
        unique_cols = []
        if col1_is_unique:
            unique_cols.append(f"{col1} ({col1_unique_ratio:.1%})")
        if col2_is_unique:
            unique_cols.append(f"{col2} ({col2_unique_ratio:.1%})")
        
        return {
            "status": "skipped", 
            "message": f"Skipped: Fully unique columns make Cramér's V meaningless - {', '.join(unique_cols)}",
            "cramers_v": None,
            "reason": "fully_unique_columns",
            "skip_details": {
                "col1_unique_ratio": col1_unique_ratio,
                "col2_unique_ratio": col2_unique_ratio,
                "col1_is_unique": col1_is_unique,
                "col2_is_unique": col2_is_unique
            }
        }
    
    # LAYER 4: Performance-optimized Cramér's V with significance testing
    return enhancer.performance_optimized_cramers_v(df, col1, col2)

def check_categorical_association_cramers_v_basic(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    # Fallback basic implementation if enhanced version unavailable
    import pandas as pd
    import numpy as np
    
    s1 = df[col1].astype(str)
    s2 = df[col2].astype(str)
    ct = pd.crosstab(s1, s2)
    n = ct.values.sum()
    if n == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"status": "skipped", "message": "Insufficient categories", "cramers_v": None}
    row_sums = ct.values.sum(axis=1, keepdims=True)
    col_sums = ct.values.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((ct.values - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    k = min(ct.shape)
    v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 and n > 0 else np.nan
    
    if not np.isnan(v):
        if v >= 0.5:
            strength = "strong"
            status = "high"
        elif v >= 0.3:
            strength = "moderate"
            status = "medium"
        else:
            strength = "weak" if v >= 0.1 else "very weak"
            status = "low"
    else:
        strength = "unknown"
        status = "low"
    
    return {
        "cramers_v": float(v) if not np.isnan(v) else None,
        "strength": strength,
        "status": status,
        "message": f"Cramér's V: {v:.3f} ({strength})" if not np.isnan(v) else "Cramér's V unavailable"
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_privacy_rules(self):
        """Add privacy and security-focused rules."""
        rules = [
            Rule(
                id="pii_pattern_detection",
                name="PII Pattern Detection",
                description="Count occurrences of common PII patterns in textual columns",
                category="privacy_security",
                column_types=["textual"],
                diversity_levels=["medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high"],
                code_template=r"""
def check_pii_pattern_detection(series: pd.Series) -> Dict[str, Any]:
    s = series.astype(str).dropna()
    if len(s) == 0:
        return {"status": "warning", "message": "No data", "pattern_counts": {}}
    
    pii_patterns = {
        'email_pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone_pattern': r'^\+?[\d\s\-\(\)]{10,}$',
        'ssn_pattern': r'^\d{3}-?\d{2}-?\d{4}$',
        'credit_card_pattern': r'^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$',
        'ip_address_pattern': r'^(?:\d{1,3}\.){3}\d{1,3}$'
    }
    
    pattern_counts = {}
    total_count = len(s)
    
    for pattern_name, pattern in pii_patterns.items():
        matches = s.str.match(pattern, na=False).sum()
        pattern_counts[pattern_name] = {
            "count": int(matches),
            "ratio": float(matches / total_count) if total_count > 0 else 0.0
        }
    
    total_pii_matches = sum(p["count"] for p in pattern_counts.values())
    
    return {
        "pattern_counts": pattern_counts,
        "total_pii_matches": total_pii_matches,
        "total_rows": total_count,
        "pii_ratio": float(total_pii_matches / total_count) if total_count > 0 else 0.0,
        "status": "passed",
        "message": f"Found {total_pii_matches} PII pattern matches in {total_count} values"
    }
"""
            ),
            
            Rule(
                id="data_format_patterns",
                name="Data Format Patterns",
                description="Identify common data format patterns in textual columns",
                category="privacy_security",
                column_types=["textual"],
                diversity_levels=["high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium"],
                code_template=r"""
def check_data_format_patterns(series: pd.Series) -> Dict[str, Any]:
    s = series.astype(str).dropna()
    if len(s) == 0:
        return {"status": "warning", "message": "No data"}
    
    # Simple format pattern detection
    format_patterns = {
        'masked_asterisk': r'\*{3,}',
        'masked_x': r'[xX]{3,}',
        'hex_string': r'^[0-9a-fA-F]{8,}$',
        'base64_like': r'^[A-Za-z0-9+/]{20,}=*$',
        'all_caps': r'^[A-Z][A-Z0-9_]*$',
        'camel_case': r'^[a-z][a-zA-Z0-9]*$',
        'snake_case': r'^[a-z][a-z0-9_]*$'
    }
    
    format_counts = {}
    total_count = len(s)
    
    for pattern_name, pattern in format_patterns.items():
        matches = s.str.match(pattern, na=False).sum()
        format_counts[pattern_name] = {
            "count": int(matches),
            "ratio": float(matches / total_count) if total_count > 0 else 0.0
        }
    
    # Find the most common format
    dominant_format = max(format_counts.keys(), 
                         key=lambda k: format_counts[k]["ratio"])
    dominant_ratio = format_counts[dominant_format]["ratio"]
    
    return {
        "format_counts": format_counts,
        "dominant_format": dominant_format,
        "dominant_format_ratio": dominant_ratio,
        "total_rows": total_count,
        "status": "passed",
        "message": f"Dominant format: {dominant_format} ({dominant_ratio:.1%} of values)"
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_advanced_cross_column_rules(self):
        """Add advanced cross-column analysis rules."""
        rules = []
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
    def _add_performance_rules(self):
        """Add performance and efficiency-focused rules."""
        rules = [
            Rule(
                id="memory_usage_analysis",
                name="Memory Usage Analysis",
                description="Analyze memory usage and suggest potential optimizations",
                category="performance",
                column_types=["numeric", "textual", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_memory_usage_analysis(series: pd.Series) -> Dict[str, Any]:
    current_dtype = str(series.dtype)
    memory_usage = series.memory_usage(deep=True)
    
    # Calculate memory per value
    total_count = len(series)
    memory_per_value = memory_usage / total_count if total_count > 0 else 0
    
    # Suggest optimizations for numeric data
    optimization_potential = None
    if pd.api.types.is_numeric_dtype(series) and not series.empty:
        if pd.api.types.is_integer_dtype(series):
            min_val = series.min()
            max_val = series.max()
            
            # Check if smaller integer types would work
            if -128 <= min_val <= 127 and -128 <= max_val <= 127:
                optimization_potential = "int8"
            elif -32768 <= min_val <= 32767 and -32768 <= max_val <= 32767:
                optimization_potential = "int16"
            elif -2147483648 <= min_val <= 2147483647:
                optimization_potential = "int32"
    
    # For object/string columns, check categorical potential
    categorical_potential = False
    if pd.api.types.is_object_dtype(series):
        unique_count = len(set(series.dropna()))
        if unique_count < total_count * 0.5:  # Less than 50% unique
            categorical_potential = True
    
    return {
        "current_dtype": current_dtype,
        "memory_usage_bytes": int(memory_usage),
        "memory_usage_mb": float(memory_usage / (1024 * 1024)),
        "memory_per_value_bytes": float(memory_per_value),
        "total_values": total_count,
        "optimization_potential": optimization_potential,
        "categorical_potential": categorical_potential,
        "status": "passed",
        "message": f"Memory: {memory_usage/1024:.1f} KB ({memory_per_value:.1f} bytes/value)"
    }
"""
            ),
            
            Rule(
                id="cardinality_analysis",
                name="Cardinality Analysis",
                description="Analyze cardinality characteristics for indexing and storage decisions",
                category="performance",
                column_types=["numeric", "textual", "categorical"],
                diversity_levels=["low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template=r"""
def check_cardinality_analysis(series: pd.Series) -> Dict[str, Any]:
    total_count = len(series)
    # Use len(set()) for better performance, excluding NaN values like nunique()
    unique_count = len(set(series.dropna()))
    null_count = series.isnull().sum()
    non_null_count = total_count - null_count
    
    # Calculate cardinality metrics
    cardinality_ratio = unique_count / non_null_count if non_null_count > 0 else 0
    
    # Classify cardinality level
    if cardinality_ratio == 1.0:
        cardinality_level = "unique"
    elif cardinality_ratio >= 0.9:
        cardinality_level = "very_high"
    elif cardinality_ratio >= 0.5:
        cardinality_level = "high"
    elif cardinality_ratio >= 0.1:
        cardinality_level = "medium"
    else:
        cardinality_level = "low"
    
    # Index suitability analysis
    index_suitability = "high" if cardinality_ratio >= 0.8 else (
        "medium" if cardinality_ratio >= 0.3 else "low"
    )
    
    # Compression potential (inverse of cardinality)
    compression_potential = "high" if cardinality_ratio < 0.1 else (
        "medium" if cardinality_ratio < 0.5 else "low"
    )
    
    return {
        "total_count": total_count,
        "unique_count": unique_count,
        "null_count": null_count,
        "cardinality_ratio": float(cardinality_ratio),
        "cardinality_level": cardinality_level,
        "index_suitability": index_suitability,
        "compression_potential": compression_potential,
        "status": "passed",
        "message": f"Cardinality: {unique_count}/{total_count} ({cardinality_ratio:.1%}), level: {cardinality_level}"
    }
"""
            )
        ]
        
        for rule in rules:
            self._register_rule(rule, builtin=True)
    
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
    

    
    def _generate_column_hash(self, series: pd.Series) -> str:
        """
        Generate a hash for column content to enable caching.
        WARNING: This can be memory-intensive for large columns.
        
        Args:
            series: Pandas Series to hash
            
        Returns:
            Hash string representing the column content
        """
        # Sample the series for large datasets to avoid memory issues
        if len(series) > 10000:
            warnings.warn(
                "Generating hash for large column (>10K rows). This may consume significant memory.",
                MemoryWarning
            )
            # Use a sample for hashing to reduce memory usage
            sample_series = series.sample(n=min(10000, len(series)), random_state=42)
        else:
            sample_series = series
            
        # Create a hash based on series content and metadata
        content_str = f"{sample_series.dtype}_{len(series)}_{sample_series.to_string()}"
        return hashlib.md5(content_str.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _get_cached_column_attributes(self, col_hash: str, col_name: str, 
                                    dtype_str: str, length: int) -> Optional[str]:
        """
        Cache column attribute calculations.
        WARNING: Large cache sizes can consume significant memory.
        
        Args:
            col_hash: Hash of the column content
            col_name: Column name
            dtype_str: String representation of data type
            length: Number of rows
            
        Returns:
            Cached attribute identifier or None if not cached
        """
        # This is a placeholder for cached attributes
        # In practice, you'd store the actual ColumnAttributes objects
        return f"cached_{col_hash}_{dtype_str}_{length}"
    
    @lru_cache(maxsize=500)
    def _is_rule_applicable_cached(self, rule_id: str, type_cat: str, 
                                 diversity: str, nullability: str) -> bool:
        """
        Cache rule applicability decisions for performance.
        
        Args:
            rule_id: Rule identifier
            type_cat: Type category string
            diversity: Diversity level string
            nullability: Nullability level string
            
        Returns:
            True if rule is applicable
        """
        rule = self.get_rule(rule_id)
        if not rule:
            return False
            
        # Check rule applicability based on attributes
        if type_cat not in rule.column_types:
            return False
        if diversity not in rule.diversity_levels:
            return False
        if nullability not in rule.nullability_levels:
            return False
            
        return True
    
    def clear_caches(self):
        """
        Clear all caches to free memory.
        Call this method if memory usage becomes a concern.
        """
        self._get_cached_column_attributes.cache_clear()
        self._is_rule_applicable_cached.cache_clear()
        warnings.warn("Rule engine caches have been cleared to free memory.", UserWarning)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cache usage for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "column_attributes_cache": self._get_cached_column_attributes.cache_info(),
            "rule_applicability_cache": self._is_rule_applicable_cached.cache_info()
        }
    
    def determine_analysis_depth(self, attributes: ColumnAttributes) -> str:
        """
        Determine appropriate analysis depth based on column characteristics.
        Allows progressive analysis - start shallow, deepen only for interesting columns.
        
        Args:
            attributes: Column attributes
            
        Returns:
            Analysis depth level: "basic", "standard", or "deep"
        """
        # Basic analysis for constant columns (no variation to analyze)
        if attributes.diversity_level == DiversityLevel.CONSTANT:
            return "basic"
        
        # Basic analysis for columns that are likely indices (already covered by other logic)
        if attributes.is_likely_index:
            return "basic"
        
        # Standard analysis is the default for most columns
        # Deep analysis reserved for columns with rich patterns worth investigating
        if (attributes.diversity_level in [DiversityLevel.HIGH, DiversityLevel.DISTINCTIVE] and
            attributes.type_category in [TypeCategory.NUMERIC, TypeCategory.TEXTUAL]):
            return "deep"
        
        return "standard"
    
    def get_rules_by_analysis_depth(self, depth: str) -> List[str]:
        """
        Get rule IDs appropriate for a specific analysis depth.
        
        Args:
            depth: Analysis depth ("basic", "standard", "deep")
            
        Returns:
            List of rule IDs appropriate for this depth
        """
        analysis_levels = {
            "basic": [
                "null_analysis", "unique_count", "most_common_value", 
                "basic_stats", "memory_usage_analysis"
            ],
            "standard": [
                # Includes basic rules plus:
                "data_type_inference", "top_k_frequencies", "parseability_analysis",
                "pii_pattern_detection", "cardinality_analysis", "length_analysis"
            ],
            "deep": [
                # Includes standard rules plus:
                "outlier_detection", "distribution_analysis", "modality_estimation",
                "outlier_detection_zscore", "numeric_histogram_quantiles",
                "text_patterns", "whitespace_encoding_checks", "stability_entropy",
                "correlation"  # For cross-column analysis
            ]
        }
        
        # Build cumulative rule list
        rules_for_depth = []
        if depth in ["basic", "standard", "deep"]:
            rules_for_depth.extend(analysis_levels["basic"])
        if depth in ["standard", "deep"]:
            rules_for_depth.extend(analysis_levels["standard"])
        if depth == "deep":
            rules_for_depth.extend(analysis_levels["deep"])
            
        return rules_for_depth
    
    def get_relevant_rules_with_depth(self, attributes: ColumnAttributes, 
                                    enable_cross_column: bool = True,
                                    force_depth: str = None) -> List[Rule]:
        """
        Get rules relevant for a column based on attributes and analysis depth.
        
        Args:
            attributes: Column attributes
            enable_cross_column: Whether to include cross-column rules
            force_depth: Override automatic depth determination
            
        Returns:
            List of relevant rules filtered by analysis depth
        """
        # Determine analysis depth
        depth = force_depth or self.determine_analysis_depth(attributes)
        appropriate_rule_ids = set(self.get_rules_by_analysis_depth(depth))
        
        # Get base relevant rules
        relevant_rules = self.get_relevant_rules(attributes, enable_cross_column)
        
        # Filter by analysis depth
        depth_filtered_rules = [
            rule for rule in relevant_rules 
            if rule.id in appropriate_rule_ids
        ]
        
        return depth_filtered_rules
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID."""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[Rule]:
        """List all available rules."""
        return list(self.rules.values())
    
    def get_rules_by_category(self, category: str) -> List[Rule]:
        """Get all rules in a specific category."""
        return [rule for rule in self.rules.values() if rule.category == category]

    # ------------------------- Compatibility helpers -------------------------
    def _register_rule(self, rule: Rule, builtin: bool = True) -> None:
        self.rules[rule.id] = rule
        # Track category membership
        self.rule_categories.setdefault(rule.category, []).append(rule.id)
        # Track dependencies
        self.rule_dependencies[rule.id] = list(rule.dependencies or [])
        # Track metadata
        self.rule_priorities[rule.id] = getattr(rule, "priority", RulePriority.MEDIUM)
        self.rule_complexities[rule.id] = getattr(rule, "complexity", RuleComplexity.MEDIUM)
        if not builtin:
            self._user_rule_ids.add(rule.id)

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine (compat API)."""
        self._register_rule(rule, builtin=False)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule from the engine (compat API)."""
        if rule_id in self.rules:
            category = self.rules[rule_id].category
            del self.rules[rule_id]
            if category in self.rule_categories and rule_id in self.rule_categories[category]:
                self.rule_categories[category].remove(rule_id)
            self.rule_dependencies.pop(rule_id, None)
            self.rule_priorities.pop(rule_id, None)
            self.rule_complexities.pop(rule_id, None)

    def get_rule_dependencies(self, rule_id: str) -> List[str]:
        return list(self.rule_dependencies.get(rule_id, []))

    def validate_rule_dependencies(self, rule_id: str) -> bool:
        deps = self.rule_dependencies.get(rule_id, [])
        # Fail on self-dependency
        if rule_id in deps:
            return False
        # All deps must exist
        for dep in deps:
            if dep not in self.rules:
                return False
        return True

    def save_rules_to_file(self, filename: str) -> None:
        # Save only user-added rules for compatibility with tests
        rules_to_save = [self.rules[rid] for rid in self._user_rule_ids if rid in self.rules]
        data = {
            "rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "category": r.category,
                    "column_types": r.column_types,
                    "diversity_levels": r.diversity_levels,
                    "nullability_levels": r.nullability_levels,
                    "requires_cross_column": r.requires_cross_column,
                    "dependencies": r.dependencies,
                    "priority": getattr(r.priority, "value", str(r.priority)),
                    "complexity": getattr(r.complexity, "value", str(r.complexity)),
                    "code_template": r.code_template,
                    "parameters": r.parameters,
                    "enabled": getattr(r, "enabled", True),
                }
                for r in rules_to_save
            ]
        }
        with open(filename, "w") as f:
            f.write(json.dumps(data))

    def load_rules_from_file(self, filename: str) -> None:
        with open(filename, "r") as f:
            data = json.load(f)
        for rd in data.get("rules", []):
            # Map strings back to enums if needed
            pr = rd.get("priority", "medium")
            cx = rd.get("complexity", "medium")
            if not isinstance(pr, RulePriority):
                pr = RulePriority(pr)
            if not isinstance(cx, RuleComplexity):
                cx = RuleComplexity(cx)
            rule = Rule(
                id=rd["id"],
                name=rd.get("name", rd["id"]),
                description=rd.get("description", ""),
                category=rd.get("category", "other"),
                column_types=rd.get("column_types", ["numeric", "textual", "temporal", "boolean", "categorical"]),
                diversity_levels=rd.get("diversity_levels", ["low", "medium", "high", "distinctive", "binary", "constant"]),
                nullability_levels=rd.get("nullability_levels", ["empty", "low", "medium", "high", "full"]),
                requires_cross_column=rd.get("requires_cross_column", False),
                dependencies=rd.get("dependencies", []),
                priority=pr,
                complexity=cx,
                code_template=rd.get("code_template", ""),
                parameters=rd.get("parameters", {}),
                enabled=rd.get("enabled", True),
            )
            self.add_rule(rule)

    def get_rule_statistics(self) -> Dict[str, Any]:
        # Compute statistics over user-added rules only
        categories: Dict[str, int] = {}
        priorities: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        complexities: Dict[str, int] = {"low": 0, "medium": 0, "high": 0}
        for rid in self._user_rule_ids:
            if rid not in self.rules:
                continue
            r = self.rules[rid]
            categories[r.category] = categories.get(r.category, 0) + 1
            p = getattr(r.priority, "value", str(r.priority))
            priorities[p] = priorities.get(p, 0) + 1
            cx = getattr(r.complexity, "value", str(r.complexity))
            complexities[cx] = complexities.get(cx, 0) + 1
        cross_column_rules = sum(1 for rid in self._user_rule_ids if rid in self.rules and self.rules[rid].category == "cross_column")
        return {
            "total_rules": len(self._user_rule_ids),
            "categories": categories,
            "priorities": priorities,
            "complexities": complexities,
            "cross_column_rules": cross_column_rules,
        }
