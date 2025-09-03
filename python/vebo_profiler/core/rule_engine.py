"""
Rule engine for managing and executing data profiling rules.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
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
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
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
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
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
            ),
            Rule(
                id="top_k_frequencies",
                name="Top-K Frequencies",
                description="Top-k most common values, frequencies, and 'other' proportion",
                category="basic_stats",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
def check_top_k_frequencies(series: pd.Series) -> Dict[str, Any]:
    k = 10
    value_counts = series.value_counts(dropna=False)
    total_count = len(series)
    top_k = value_counts.head(k)
    other_count = max(total_count - int(top_k.sum()), 0)
    other_ratio = other_count / total_count if total_count > 0 else 0
    top_values = [
        {"value": v, "count": int(c), "ratio": (int(c) / total_count if total_count > 0 else 0)}
        for v, c in zip(top_k.index.tolist(), top_k.values.tolist())
    ]
    return {
        "top_k": top_values,
        "other_count": other_count,
        "other_ratio": other_ratio,
        "status": "passed",
        "message": f"Computed top-{k} frequencies"
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
                code_template="""
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
                code_template="""
def check_parseability_analysis(series: pd.Series) -> Dict[str, Any]:
    import json as _json
    s = series.dropna()
    total = len(s)
    if total == 0:
        return {"status": "warning", "message": "No non-null values", "parseable_int_ratio": 0, "parseable_float_ratio": 0, "parseable_datetime_ratio": 0, "parseable_json_ratio": 0}
    as_numeric = pd.to_numeric(s, errors='coerce')
    int_mask = as_numeric.notna() & (as_numeric % 1 == 0)
    float_mask = as_numeric.notna()
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
                code_template="""
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
            ),
            Rule(
                id="outlier_detection_zscore",
                name="Outlier Detection (Z-Score)",
                description="Detect outliers using Z-score method",
                category="outlier_detection",
                column_types=["numeric"],
                diversity_levels=["medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
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
                code_template="""
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
                code_template="""
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
                id="whitespace_encoding_checks",
                name="Whitespace and Encoding Checks",
                description="Leading/trailing spaces, whitespace-only, non-printable characters",
                category="text_quality",
                column_types=["textual"],
                diversity_levels=["constant", "binary", "low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                code_template="""
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
                code_template="""
def check_missingness_relationships(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    s1 = df[col1]
    s2 = df[col2]
    n = len(df)
    both_null = (s1.isna() & s2.isna()).sum()
    s1_null_only = (s1.isna() & s2.notna()).sum()
    s2_null_only = (s2.isna() & s1.notna()).sum()
    co_null_ratio = both_null / n if n > 0 else 0
    return {
        "both_null_count": int(both_null),
        "s1_null_only_count": int(s1_null_only),
        "s2_null_only_count": int(s2_null_only),
        "co_null_ratio": float(co_null_ratio),
        "status": "passed",
        "message": "Missingness relationship computed"
    }
"""
            ),
            Rule(
                id="functional_dependency",
                name="Functional Dependency Approximation",
                description="Approximate whether col1 determines col2",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template="""
def check_functional_dependency(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    g = df[[col1, col2]].dropna().groupby(col1, dropna=False)[col2].nunique()
    if len(g) == 0:
        return {"status": "warning", "message": "Insufficient data", "fd_holds_ratio": 0}
    violating = (g > 1).sum()
    fd_holds_ratio = 1.0 - (violating / len(g))
    status = "passed" if fd_holds_ratio >= 0.95 else ("warning" if fd_holds_ratio >= 0.8 else "failed")
    return {
        "fd_holds_ratio": float(fd_holds_ratio),
        "violating_groups": int(violating),
        "total_groups": int(len(g)),
        "status": status,
        "message": f"FD holds in {fd_holds_ratio:.2%} of groups"
    }
"""
            ),
            Rule(
                id="composite_uniqueness",
                name="Composite Uniqueness",
                description="Whether a combination of two columns yields unique rows",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template="""
def check_composite_uniqueness(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    dup_count = df.duplicated(subset=[col1, col2]).sum()
    total = len(df)
    unique_ratio = 1.0 - (dup_count / total if total > 0 else 0)
    status = "passed" if dup_count == 0 else ("warning" if unique_ratio >= 0.99 else "failed")
    return {
        "duplicate_pairs": int(dup_count),
        "unique_pair_ratio": float(unique_ratio),
        "status": status,
        "message": f"Unique pair ratio: {unique_ratio:.2%}"
    }
"""
            ),
            Rule(
                id="inclusion_dependency",
                name="Inclusion Dependency",
                description="Whether values of col1 appear as a subset of col2",
                category="cross_column",
                column_types=["numeric", "textual", "temporal", "boolean", "categorical"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template="""
def check_inclusion_dependency(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    s1 = pd.Series(df[col1].dropna().unique())
    s2_set = set(pd.Series(df[col2].dropna().unique()).tolist())
    if len(s1) == 0:
        return {"status": "warning", "message": "No values in first column", "coverage_ratio": 0}
    in_set = s1.apply(lambda x: x in s2_set)
    coverage_ratio = float(in_set.sum()) / float(len(s1))
    status = "passed" if coverage_ratio == 1.0 else ("warning" if coverage_ratio >= 0.9 else "failed")
    return {
        "coverage_ratio": coverage_ratio,
        "status": status,
        "message": f"{coverage_ratio:.2%} of {col1} values appear in {col2}"
    }
"""
            ),
            Rule(
                id="categorical_association_cramers_v",
                name="Categorical Association (Cramér’s V)",
                description="Association strength between two categorical columns",
                category="cross_column",
                column_types=["categorical", "textual"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium", "high", "full"],
                requires_cross_column=True,
                code_template="""
def check_categorical_association_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    s1 = df[col1].astype(str)
    s2 = df[col2].astype(str)
    ct = pd.crosstab(s1, s2)
    n = ct.values.sum()
    if n == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"status": "warning", "message": "Insufficient categories", "cramers_v": None}
    row_sums = ct.values.sum(axis=1, keepdims=True)
    col_sums = ct.values.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((ct.values - expected) ** 2 / np.where(expected == 0, np.nan, expected))
    k = min(ct.shape)
    v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 and n > 0 else np.nan
    strength = (
        "strong" if v >= 0.5 else ("moderate" if v >= 0.3 else ("weak" if v >= 0.1 else "very weak"))
    ) if not np.isnan(v) else "unknown"
    return {
        "cramers_v": float(v) if not np.isnan(v) else None,
        "strength": strength,
        "status": "passed",
        "message": f"Cramér’s V: {v:.3f}" if not np.isnan(v) else "Cramér’s V unavailable"
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
                code_template="""
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
                code_template="""
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
        rules = [
            Rule(
                id="column_name_similarity",
                name="Column Name Similarity",
                description="Calculate similarity between column names to identify relationships",
                category="advanced_cross_column",
                column_types=["numeric", "textual", "temporal", "categorical"],
                diversity_levels=["low", "medium", "high", "distinctive", "fully_unique"],
                nullability_levels=["empty", "low", "medium", "high"],
                requires_cross_column=True,
                code_template="""
def check_column_name_similarity(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    import difflib
    
    # Calculate name similarity
    name_similarity = difflib.SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
    
    # Check for common prefixes/suffixes
    col1_lower = col1.lower()
    col2_lower = col2.lower()
    
    common_prefix_len = 0
    for i, (c1, c2) in enumerate(zip(col1_lower, col2_lower)):
        if c1 == c2:
            common_prefix_len = i + 1
        else:
            break
    
    common_suffix_len = 0
    for i, (c1, c2) in enumerate(zip(reversed(col1_lower), reversed(col2_lower))):
        if c1 == c2:
            common_suffix_len = i + 1
        else:
            break
    
    # Check for semantic relationships
    relationship_indicators = {
        'start_end': ('start' in col1_lower and 'end' in col2_lower) or 
                    ('end' in col1_lower and 'start' in col2_lower),
        'min_max': ('min' in col1_lower and 'max' in col2_lower) or 
                  ('max' in col1_lower and 'min' in col2_lower),
        'first_last': ('first' in col1_lower and 'last' in col2_lower) or 
                     ('last' in col1_lower and 'first' in col2_lower)
    }
    
    semantic_relationship = None
    for rel_type, has_relationship in relationship_indicators.items():
        if has_relationship:
            semantic_relationship = rel_type
            break
    
    return {
        "name_similarity": float(name_similarity),
        "common_prefix_length": common_prefix_len,
        "common_suffix_length": common_suffix_len,
        "semantic_relationship": semantic_relationship,
        "col1_name": col1,
        "col2_name": col2,
        "status": "passed",
        "message": f"Name similarity: {name_similarity:.2f}, semantic: {semantic_relationship or 'none'}"
    }
"""
            ),
            
            Rule(
                id="value_overlap_analysis",
                name="Value Overlap Analysis", 
                description="Analyze overlap of values between two columns",
                category="advanced_cross_column",
                column_types=["textual", "categorical", "numeric"],
                diversity_levels=["low", "medium", "high", "distinctive"],
                nullability_levels=["empty", "low", "medium"],
                requires_cross_column=True,
                code_template="""
def check_value_overlap_analysis(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    s1 = df[col1].dropna()
    s2 = df[col2].dropna()
    
    if len(s1) == 0 or len(s2) == 0:
        return {"status": "warning", "message": "No data in one or both columns"}
    
    # Convert to sets for overlap analysis
    set1 = set(s1.astype(str).unique())
    set2 = set(s2.astype(str).unique())
    
    # Calculate overlap statistics
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    overlap_count = len(intersection)
    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0
    
    # Overlap ratios for each column
    col1_overlap_ratio = overlap_count / len(set1) if len(set1) > 0 else 0
    col2_overlap_ratio = overlap_count / len(set2) if len(set2) > 0 else 0
    
    return {
        "col1_unique_values": len(set1),
        "col2_unique_values": len(set2),
        "overlapping_values": overlap_count,
        "jaccard_similarity": float(jaccard_similarity),
        "col1_overlap_ratio": float(col1_overlap_ratio),
        "col2_overlap_ratio": float(col2_overlap_ratio),
        "total_unique_values": len(union),
        "status": "passed",
        "message": f"Overlap: {overlap_count} values, Jaccard: {jaccard_similarity:.3f}"
    }
"""
            )
        ]
        
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
                code_template="""
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
        unique_count = series.nunique()
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
                code_template="""
def check_cardinality_analysis(series: pd.Series) -> Dict[str, Any]:
    total_count = len(series)
    unique_count = series.nunique()
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
            
            # Skip unnecessary rules for likely index columns
            if self._should_skip_rule_for_index(rule, attributes):
                continue
            
            relevant_rules.append(rule)
        
        return relevant_rules
    
    def _should_skip_rule_for_index(self, rule: Rule, attributes: ColumnAttributes) -> bool:
        """
        Determine if a rule should be skipped for likely index columns.
        
        Args:
            rule: Rule to check
            attributes: Column attributes
            
        Returns:
            True if the rule should be skipped for this index column
        """
        # Only skip rules for likely index columns
        if not attributes.is_likely_index:
            return False
        
        # Rules that are typically irrelevant for index columns
        irrelevant_for_index = {
            # Statistical analysis rules
            "numeric_stats", "outlier_detection", "modality_estimation", 
            "numeric_histogram_quantiles", "outlier_detection_zscore",
            
            # Text pattern analysis (indices are usually not meaningful patterns)
            "text_patterns", "whitespace_encoding_checks",
            
            # Analysis that's not useful for unique identifiers
            "stability_entropy",  # All values are unique, so entropy is max
            "duplicate_value_analysis",  # No duplicates by definition
            
            # Length analysis might not be relevant for IDs (though could be kept)
            "length_analysis"
        }
        
        # Keep basic stats and uniqueness analysis even for indices
        essential_for_index = {
            "unique_count", "most_common_value", "null_analysis", 
            "top_k_frequencies", "parseability_analysis",
            # Privacy rules are important for index columns (PII detection)
            "pii_pattern_detection", "data_format_patterns",
            # Performance rules are useful for index columns
            "memory_usage_analysis", "cardinality_analysis"
        }
        
        # Skip irrelevant rules
        if rule.id in irrelevant_for_index:
            return True
        
        # Keep essential rules
        if rule.id in essential_for_index:
            return False
        
        # For rules not explicitly categorized, use category-based logic
        if rule.category in ["outlier_detection", "text_quality", "distribution_analysis"]:
            return True
        
        return False
    
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
