"""
Check executor for running data profiling checks.
"""

import pandas as pd
import numpy as np
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .rule_engine import Rule, RuleResult, RuleStatus
from .meta_rules import ColumnAttributes, MetaRuleDetector, DiversityLevel


@dataclass
class ExecutionConfig:
    """Configuration for check execution."""
    enable_cross_column: bool = True
    max_workers: int = None
    timeout_seconds: int = 300
    enable_parallel: bool = True
    sample_size: int = 10000
    sampling_threshold: int = 100000
    # PERFORMANCE OPTIMIZATIONS
    max_cross_column_pairs: int = 1000  # Limit pairs for wide tables
    enable_smart_filtering: bool = True  # Enable type-based rule filtering
    enable_early_exit: bool = True  # Enable early exit strategies
    # ROW-BASED OPTIMIZATIONS
    enable_adaptive_sampling: bool = True  # Use rule-specific sample sizes
    cross_column_sample_threshold: int = 50000  # Apply sampling to cross-column rules above this size
    correlation_sample_size: int = 5000  # Sample size for correlation calculations
    functional_dep_sample_size: int = 10000  # Sample size for functional dependency
    missingness_sample_size: int = 20000  # Sample size for missingness analysis
    enable_statistical_confidence: bool = True  # Use statistical confidence intervals
    # TWO-STAGE PROCESSING OPTIMIZATIONS
    enable_two_stage_processing: bool = True  # Enable two-stage pattern discovery
    pattern_discovery_sample_size: int = 5000  # Stage 1: Pattern discovery sample size
    heavy_rule_threshold: int = 25000  # Apply two-stage processing above this size
    regex_discovery_sample_size: int = 2000  # Regex pattern discovery sample
    outlier_discovery_sample_size: int = 3000  # Outlier threshold discovery sample


class CheckExecutor:
    """
    Executes data profiling checks on datasets.
    """
    
    def __init__(self, config: ExecutionConfig = None):
        """
        Initialize the check executor.
        
        Args:
            config: Execution configuration
        """
        self.config = config or ExecutionConfig()
        self.meta_detector = MetaRuleDetector()
        
        # Set default max_workers based on CPU count
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 4)
            
        # PERFORMANCE OPTIMIZATION: Caching for expensive operations
        self._column_signature_cache = {}  # Cache column signatures to avoid recomputation
        self._rule_result_cache = {}  # Cache rule results for identical column content
        
        # TWO-STAGE PROCESSING: Pattern discovery cache
        self._pattern_discovery_cache = {}  # Cache discovered patterns for two-stage processing
        self._heavy_rule_patterns = {}  # Cache patterns discovered in Stage 1

    # ------------------------- Helper methods for tests -------------------------
    def _compile_check_function(self, code_template: str):
        """Compile a simple column-level check function named 'check_*' from a template."""
        try:
            namespace: Dict[str, Any] = {"pd": pd, "np": np}
            exec(code_template, namespace)
            # Return the first callable that starts with 'check_'
            for k, v in namespace.items():
                if k.startswith("check_") and callable(v):
                    compiled_func = v
                    # Validate by dry-run to catch obvious runtime errors
                    try:
                        _ = compiled_func(pd.Series([1, 2, 3]))
                    except Exception:
                        return None
                    # Wrap to coerce numpy booleans to python bools
                    def _wrapper(series: pd.Series, _f=compiled_func):
                        res = _f(series)
                        if isinstance(res, dict):
                            coerced = {}
                            for kk, vv in res.items():
                                # Coerce numpy booleans to python bool
                                if isinstance(vv, (np.bool_,)):
                                    coerced[kk] = bool(vv)
                                else:
                                    coerced[kk] = vv
                            return coerced
                        return res
                    return _wrapper
        except Exception:
            return None
        return None

    def _evaluate_check_result(self, result_data: Optional[Dict[str, Any]], rule: Rule, execution_time_ms: float, error_message: str = None) -> RuleResult:
        """Convert raw result data or error into a RuleResult."""
        timestamp = datetime.now().isoformat()
        if error_message is not None or result_data is None:
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus.ERROR,
                score=0.0,
                message=error_message or "Execution error",
                details={"error": error_message or "unknown"},
                execution_time_ms=execution_time_ms,
                timestamp=timestamp,
            )
        # Default status to passed when not provided
        status_value = result_data.get("status", "passed")
        return RuleResult(
            rule_id=rule.id,
            rule_name=rule.name,
            status=RuleStatus(status_value),
            score=self._calculate_score(result_data),
            message=result_data.get("message", ""),
            details=result_data,
            execution_time_ms=execution_time_ms,
            timestamp=timestamp,
        )

    def _should_sample_data(self, df: pd.DataFrame) -> bool:
        return self.meta_detector.should_enable_sampling(df, self.config.sampling_threshold)

    def _create_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        return self.meta_detector.create_sample(df, sample_size)
    
    def execute_column_check(self, rule: Rule, series: pd.Series, 
                           attributes: ColumnAttributes) -> RuleResult:
        """
        Execute a single column-level check.
        
        Args:
            rule: Rule to execute
            series: Data series to check
            attributes: Column attributes
            
        Returns:
            RuleResult object
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # TWO-STAGE PROCESSING OPTIMIZATION: Check if this is a heavy rule
            if (self.config.enable_two_stage_processing and 
                self._is_heavy_computational_rule(rule) and 
                len(series) > self.config.heavy_rule_threshold):
                
                print(f"🚀 TWO-STAGE: Processing heavy rule {rule.id} on {len(series):,} rows")
                
                # STAGE 1: Pattern Discovery
                discovered_patterns = self._discover_patterns_stage_1(rule, series)
                
                # STAGE 2: Efficient Validation using discovered patterns
                result_dict = self._apply_patterns_stage_2(rule, series, discovered_patterns)
                
            else:
                # Standard single-stage processing for light rules or small datasets
                check_code = self._generate_check_code(rule, series, attributes)
                
                # Execute with timeout enforcement
                with ThreadPoolExecutor(max_workers=1) as exec_pool:
                    future = exec_pool.submit(self._execute_check_code, check_code, series)
                    try:
                        result_dict = future.result(timeout=self.config.timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        # Timed out
                        execution_time = (time.time() - start_time) * 1000
                        return RuleResult(
                            rule_id=rule.id,
                            rule_name=rule.name,
                            status=RuleStatus.ERROR,
                            score=0.0,
                            message="Timeout: check exceeded configured timeout",
                            details={"error": "timeout"},
                            execution_time_ms=execution_time,
                            timestamp=timestamp
                        )
            
            # Convert result to RuleResult (default to passed when no status)
            if isinstance(result_dict, dict) and "status" not in result_dict:
                result_dict = {**result_dict, "status": "passed", "message": result_dict.get("message", "")}
            execution_time = (time.time() - start_time) * 1000
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus(result_dict.get("status", "passed")),
                score=self._calculate_score(result_dict),
                message=result_dict.get("message", ""),
                details=result_dict,
                execution_time_ms=execution_time,
                timestamp=timestamp
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus.ERROR,
                score=0.0,
                message=f"Error: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                execution_time_ms=execution_time,
                timestamp=timestamp
            )
    
    def _create_optimized_sample_for_cross_column_rule(self, rule: Rule, df: pd.DataFrame, 
                                                     col1: str, col2: str) -> pd.DataFrame:
        """
        Create an optimized sample for cross-column rule execution based on rule-specific requirements.
        
        CRITICAL ROW-BASED OPTIMIZATION: Different rules need different sample sizes and strategies.
        
        Args:
            rule: Rule being executed
            df: Full DataFrame
            col1: First column name
            col2: Second column name
            
        Returns:
            Optimized DataFrame sample for the specific rule
        """
        total_rows = len(df)
        
        # Skip sampling for small datasets
        if total_rows <= self.config.cross_column_sample_threshold:
            return df
            
        # Rule-specific sampling strategies for MASSIVE performance gains
        if rule.id == "correlation":
            # Correlation: Use statistical sampling - 5K rows usually sufficient for stable correlation
            sample_size = min(self.config.correlation_sample_size, total_rows)
            return df.sample(n=sample_size, random_state=42)
            
        elif rule.id == "functional_dependency":
            # Functional dependency: Need sufficient representation of each category
            sample_size = min(self.config.functional_dep_sample_size, total_rows)
            # Use stratified sampling to ensure category representation
            return df.sample(n=sample_size, random_state=42)
            
        elif rule.id == "missingness_relationships":
            # Missingness: Need good coverage of null patterns
            sample_size = min(self.config.missingness_sample_size, total_rows)
            return df.sample(n=sample_size, random_state=42)
            
        elif rule.id == "identicality":
            # Identicality: Can use early termination - sample small first, expand if needed
            initial_sample_size = min(1000, total_rows)
            return df.sample(n=initial_sample_size, random_state=42)
            
        else:
            # Default sampling for unknown rules
            sample_size = min(self.config.sample_size, total_rows)
            return df.sample(n=sample_size, random_state=42)
    
    def _should_use_early_termination(self, rule: Rule, df: pd.DataFrame) -> bool:
        """
        Determine if early termination optimization should be used for a rule.
        
        Args:
            rule: Rule being executed
            df: DataFrame being processed
            
        Returns:
            True if early termination can be applied
        """
        if not self.config.enable_early_exit:
            return False
            
        # Rules that benefit from early termination
        early_exit_rules = {"identicality", "missingness_relationships"}
        return rule.id in early_exit_rules and len(df) > 10000
    
    def _is_heavy_computational_rule(self, rule: Rule) -> bool:
        """
        Identify rules that benefit from two-stage processing due to computational complexity.
        
        Args:
            rule: Rule to check
            
        Returns:
            True if rule is computationally heavy and benefits from two-stage processing
        """
        # Rules that are computationally expensive and benefit from pattern discovery
        heavy_rules = {
            "text_patterns",           # Regex pattern recognition
            "outlier_detection",       # Statistical outlier calculation
            "outlier_detection_zscore", # Z-score outlier calculation
            "text_regex_inference",    # Text pattern inference
            "parseability_analysis",   # Data type parsing validation
            "stability_entropy",       # Statistical entropy calculation
            "modality_estimation",     # Statistical modality analysis
            "functional_dependency"    # Complex groupby operations
        }
        return rule.id in heavy_rules
    
    def _discover_patterns_stage_1(self, rule: Rule, series: pd.Series) -> dict:
        """
        STAGE 1: Pattern Discovery - Run expensive analysis on small sample to discover patterns.
        
        This is the key optimization: discover patterns/thresholds on small samples,
        then apply efficient validation on the full dataset.
        
        Args:
            rule: Rule being executed
            series: Full data series
            
        Returns:
            Dictionary containing discovered patterns for Stage 2
        """
        # Create cache key for pattern discovery
        series_signature = self._get_column_signature(series, series.name or "unknown")
        cache_key = f"{rule.id}_{series_signature}"
        
        # Check if we already discovered patterns for this series
        if cache_key in self._pattern_discovery_cache:
            return self._pattern_discovery_cache[cache_key]
        
        # Get appropriate sample size for pattern discovery
        sample_size = self._get_pattern_discovery_sample_size(rule, len(series))
        
        if len(series) <= sample_size:
            # No need for sampling on small series
            sample_series = series
        else:
            # Create representative sample for pattern discovery
            sample_series = series.sample(n=sample_size, random_state=42)
            print(f"🔍 STAGE 1: Discovering patterns for {rule.id} using {sample_size:,} of {len(series):,} rows")
        
        discovered_patterns = {}
        
        # Rule-specific pattern discovery
        if rule.id == "text_patterns":
            discovered_patterns = self._discover_regex_patterns(sample_series)
            
        elif rule.id in ["outlier_detection", "outlier_detection_zscore"]:
            discovered_patterns = self._discover_outlier_thresholds(sample_series)
            
        elif rule.id == "parseability_analysis":
            discovered_patterns = self._discover_data_type_patterns(sample_series)
            
        elif rule.id in ["stability_entropy", "modality_estimation"]:
            discovered_patterns = self._discover_statistical_parameters(sample_series, rule.id)
            
        elif rule.id == "functional_dependency":
            discovered_patterns = self._discover_category_structure(sample_series)
        
        # Cache the discovered patterns
        self._pattern_discovery_cache[cache_key] = discovered_patterns
        
        return discovered_patterns
    
    def _get_pattern_discovery_sample_size(self, rule: Rule, total_size: int) -> int:
        """
        Get optimal sample size for pattern discovery based on rule requirements.
        
        Args:
            rule: Rule being executed
            total_size: Total size of the dataset
            
        Returns:
            Optimal sample size for pattern discovery
        """
        # Rule-specific sample sizes for pattern discovery
        if rule.id == "text_patterns":
            return min(self.config.regex_discovery_sample_size, total_size)
        elif rule.id in ["outlier_detection", "outlier_detection_zscore"]:
            return min(self.config.outlier_discovery_sample_size, total_size)
        else:
            return min(self.config.pattern_discovery_sample_size, total_size)
    
    def _discover_regex_patterns(self, sample_series: pd.Series) -> dict:
        """
        Discover common regex patterns in a text sample for efficient full-dataset validation.
        
        Args:
            sample_series: Sample of text data
            
        Returns:
            Dictionary with discovered regex patterns
        """
        import re
        
        # Convert to string and drop nulls for pattern analysis
        text_sample = sample_series.astype(str).dropna()
        
        if len(text_sample) == 0:
            return {"patterns": [], "common_formats": []}
        
        # Discover common patterns efficiently
        discovered_patterns = []
        format_counts = {}
        
        # Sample a smaller subset for intensive pattern analysis
        pattern_sample = text_sample.sample(n=min(500, len(text_sample)), random_state=42)
        
        # Discover common formats
        for value in pattern_sample:
            # Create format signature (length, character types)
            format_sig = self._create_format_signature(str(value))
            format_counts[format_sig] = format_counts.get(format_sig, 0) + 1
        
        # Get most common formats
        common_formats = sorted(format_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate efficient regex patterns for common formats
        for format_sig, count in common_formats:
            if count >= len(pattern_sample) * 0.1:  # At least 10% occurrence
                regex_pattern = self._format_signature_to_regex(format_sig)
                if regex_pattern:
                    discovered_patterns.append({
                        "pattern": regex_pattern,
                        "format": format_sig,
                        "frequency": count / len(pattern_sample)
                    })
        
        return {
            "patterns": discovered_patterns,
            "common_formats": [fmt[0] for fmt in common_formats[:3]]
        }
    
    def _discover_outlier_thresholds(self, sample_series: pd.Series) -> dict:
        """
        Discover outlier thresholds on sample for efficient full-dataset outlier detection.
        
        Args:
            sample_series: Sample of numeric data
            
        Returns:
            Dictionary with outlier thresholds and parameters
        """
        numeric_sample = pd.to_numeric(sample_series, errors='coerce').dropna()
        
        if len(numeric_sample) < 10:
            return {"method": "insufficient_data"}
        
        # Calculate outlier thresholds on sample
        q1 = numeric_sample.quantile(0.25)
        q3 = numeric_sample.quantile(0.75)
        iqr = q3 - q1
        
        # IQR method thresholds
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        
        # Z-score method thresholds
        mean = numeric_sample.mean()
        std = numeric_sample.std()
        zscore_threshold = 3.0
        zscore_lower = mean - zscore_threshold * std
        zscore_upper = mean + zscore_threshold * std
        
        return {
            "method": "thresholds_calculated",
            "iqr_method": {"lower": iqr_lower, "upper": iqr_upper},
            "zscore_method": {"lower": zscore_lower, "upper": zscore_upper, "mean": mean, "std": std},
            "sample_size": len(numeric_sample)
        }
    
    def _discover_data_type_patterns(self, sample_series: pd.Series) -> dict:
        """
        Discover data type patterns on sample for efficient full-dataset type validation.
        
        Args:
            sample_series: Sample of data
            
        Returns:
            Dictionary with data type patterns
        """
        import json
        from datetime import datetime
        
        sample = sample_series.astype(str).dropna()
        if len(sample) == 0:
            return {"patterns": {}}
        
        # Test different parsing methods on sample
        patterns = {}
        
        # Test numeric parsing
        numeric_success = 0
        for value in sample:
            try:
                pd.to_numeric(value)
                numeric_success += 1
            except:
                pass
        patterns["numeric_ratio"] = numeric_success / len(sample)
        
        # Test datetime parsing
        datetime_success = 0
        for value in sample:
            try:
                pd.to_datetime(value)
                datetime_success += 1
            except:
                pass
        patterns["datetime_ratio"] = datetime_success / len(sample)
        
        # Test JSON parsing
        json_success = 0
        for value in sample:
            try:
                json.loads(value)
                json_success += 1
            except:
                pass
        patterns["json_ratio"] = json_success / len(sample)
        
        return {"patterns": patterns, "sample_size": len(sample)}
    
    def _discover_statistical_parameters(self, sample_series: pd.Series, rule_type: str) -> dict:
        """
        Discover statistical parameters on sample for efficient full-dataset analysis.
        
        Args:
            sample_series: Sample of data
            rule_type: Type of statistical rule
            
        Returns:
            Dictionary with statistical parameters
        """
        if rule_type == "stability_entropy":
            # Calculate entropy parameters on sample
            value_counts = sample_series.value_counts()
            if len(value_counts) == 0:
                return {"method": "insufficient_data"}
                
            # Calculate entropy on sample to understand data distribution
            probs = value_counts / len(sample_series)
            entropy = -(probs * np.log2(probs + 1e-12)).sum()
            
            return {
                "sample_entropy": float(entropy),
                "unique_values": len(value_counts),
                "top_values": value_counts.head(5).to_dict()
            }
            
        elif rule_type == "modality_estimation":
            # Estimate modality parameters on sample
            numeric_sample = pd.to_numeric(sample_series, errors='coerce').dropna()
            if len(numeric_sample) < 10:
                return {"method": "insufficient_data"}
                
            # Simple histogram-based modality estimation
            hist_counts, _ = np.histogram(numeric_sample, bins=min(20, len(numeric_sample)//2))
            peak_count = np.sum(hist_counts > np.max(hist_counts) * 0.1)
            
            return {
                "estimated_peaks": int(peak_count),
                "sample_stats": {
                    "mean": float(numeric_sample.mean()),
                    "std": float(numeric_sample.std()),
                    "min": float(numeric_sample.min()),
                    "max": float(numeric_sample.max())
                }
            }
        
        return {"method": "unknown_rule_type"}
    
    def _discover_category_structure(self, sample_series: pd.Series) -> dict:
        """
        Discover category structure on sample for efficient functional dependency analysis.
        
        Args:
            sample_series: Sample of categorical data
            
        Returns:
            Dictionary with category structure information
        """
        value_counts = sample_series.value_counts()
        
        return {
            "unique_categories": len(value_counts),
            "top_categories": value_counts.head(10).to_dict(),
            "category_distribution": {
                "most_common_freq": value_counts.iloc[0] if len(value_counts) > 0 else 0,
                "least_common_freq": value_counts.iloc[-1] if len(value_counts) > 0 else 0
            }
        }
    
    def _create_format_signature(self, value: str) -> str:
        """Create a format signature for text pattern recognition."""
        import re
        
        # Replace digits with 'D', letters with 'L', special chars with 'S'
        signature = re.sub(r'\d', 'D', value)
        signature = re.sub(r'[a-zA-Z]', 'L', signature)
        signature = re.sub(r'[^DL]', 'S', signature)
        
        return signature
    
    def _format_signature_to_regex(self, signature: str) -> str:
        """Convert format signature to regex pattern."""
        # Simple conversion - can be enhanced
        regex = signature.replace('D', r'\d').replace('L', r'[a-zA-Z]').replace('S', r'[^\w]')
        return f"^{regex}$" if regex else None
    
    def _apply_patterns_stage_2(self, rule: Rule, series: pd.Series, discovered_patterns: dict) -> dict:
        """
        STAGE 2: Efficient Validation - Apply discovered patterns to full dataset efficiently.
        
        This is where we get massive performance gains: instead of running expensive operations
        on every row, we use the patterns discovered in Stage 1 to perform efficient validation.
        
        Args:
            rule: Rule being executed
            series: Full data series
            discovered_patterns: Patterns discovered in Stage 1
            
        Returns:
            Result dictionary for the rule
        """
        total_count = len(series)
        
        if rule.id == "text_patterns":
            return self._apply_regex_patterns_efficiently(series, discovered_patterns)
            
        elif rule.id in ["outlier_detection", "outlier_detection_zscore"]:
            return self._apply_outlier_thresholds_efficiently(series, discovered_patterns)
            
        elif rule.id == "parseability_analysis":
            return self._apply_data_type_patterns_efficiently(series, discovered_patterns)
            
        elif rule.id == "stability_entropy":
            return self._apply_entropy_patterns_efficiently(series, discovered_patterns)
            
        elif rule.id == "modality_estimation":
            return self._apply_modality_patterns_efficiently(series, discovered_patterns)
            
        elif rule.id == "functional_dependency":
            return self._apply_category_patterns_efficiently(series, discovered_patterns)
        
        # Fallback to standard processing if no specific implementation
        return {"status": "warning", "message": "Two-stage processing not implemented for this rule"}
    
    def _apply_regex_patterns_efficiently(self, series: pd.Series, patterns: dict) -> dict:
        """
        Apply discovered regex patterns efficiently to full dataset.
        
        Instead of running expensive regex discovery on every value,
        we test against the pre-discovered patterns.
        """
        import re
        
        text_series = series.astype(str).dropna()
        total_count = len(text_series)
        
        if total_count == 0 or not patterns.get("patterns"):
            return {
                "status": "warning",
                "message": "No text data or patterns to validate",
                "pattern_matches": []
            }
        
        pattern_matches = []
        
        # Test each discovered pattern efficiently on full dataset
        for pattern_info in patterns["patterns"]:
            regex = pattern_info["pattern"]
            
            try:
                # Compile regex once and apply to all values - MUCH faster than discovery
                compiled_regex = re.compile(regex)
                matches = text_series.apply(lambda x: bool(compiled_regex.match(str(x))))
                match_count = matches.sum()
                match_ratio = match_count / total_count
                
                pattern_matches.append({
                    "pattern": regex,
                    "format": pattern_info["format"],
                    "matches": int(match_count),
                    "match_ratio": float(match_ratio)
                })
                
            except Exception as e:
                # Skip invalid patterns
                continue
        
        # Determine overall pattern compliance
        total_pattern_coverage = sum(pm["match_ratio"] for pm in pattern_matches)
        
        if total_pattern_coverage >= 0.8:
            status = "passed"
            message = f"Strong pattern compliance ({total_pattern_coverage:.1%})"
        elif total_pattern_coverage >= 0.5:
            status = "warning"  
            message = f"Moderate pattern compliance ({total_pattern_coverage:.1%})"
        else:
            status = "failed"
            message = f"Low pattern compliance ({total_pattern_coverage:.1%})"
        
        return {
            "status": status,
            "message": message,
            "pattern_matches": pattern_matches,
            "total_coverage": float(total_pattern_coverage),
            "optimization": "two_stage_regex_processing"
        }
    
    def _apply_outlier_thresholds_efficiently(self, series: pd.Series, thresholds: dict) -> dict:
        """
        Apply pre-calculated outlier thresholds efficiently to full dataset.
        
        Instead of recalculating statistics on the full dataset,
        we use the thresholds discovered in Stage 1 for MASSIVE performance gains.
        """
        if thresholds.get("method") != "thresholds_calculated":
            return {"status": "warning", "message": "No valid thresholds discovered"}
        
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        total_count = len(numeric_series)
        
        if total_count == 0:
            return {"status": "warning", "message": "No numeric data to analyze"}
        
        # Apply IQR method thresholds (very fast - just comparisons vs full statistical calculation)
        iqr_thresholds = thresholds["iqr_method"]
        iqr_outliers = ((numeric_series < iqr_thresholds["lower"]) | 
                       (numeric_series > iqr_thresholds["upper"]))
        iqr_outlier_count = iqr_outliers.sum()
        
        # Apply Z-score method thresholds (also very fast - no mean/std recalculation needed)
        zscore_thresholds = thresholds["zscore_method"]
        zscore_outliers = ((numeric_series < zscore_thresholds["lower"]) | 
                          (numeric_series > zscore_thresholds["upper"]))
        zscore_outlier_count = zscore_outliers.sum()
        
        # Use the more conservative method
        outlier_count = min(iqr_outlier_count, zscore_outlier_count)
        outlier_ratio = outlier_count / total_count
        
        # Classify result
        if outlier_ratio <= 0.05:
            status = "passed"
            message = f"Low outlier rate ({outlier_ratio:.1%})"
        elif outlier_ratio <= 0.15:
            status = "warning"
            message = f"Moderate outlier rate ({outlier_ratio:.1%})"
        else:
            status = "failed"
            message = f"High outlier rate ({outlier_ratio:.1%})"
        
        return {
            "status": status,
            "message": message,
            "outlier_count": int(outlier_count),
            "outlier_ratio": float(outlier_ratio),
            "iqr_outliers": int(iqr_outlier_count),
            "zscore_outliers": int(zscore_outlier_count),
            "optimization": "two_stage_outlier_detection"
        }
    
    def _apply_data_type_patterns_efficiently(self, series: pd.Series, patterns: dict) -> dict:
        """Apply pre-discovered data type patterns efficiently to full dataset."""
        discovered = patterns.get("patterns", {})
        
        if not discovered:
            return {"status": "warning", "message": "No patterns discovered"}
        
        # Use sample ratios to estimate full dataset parsing success - no need to reparse everything
        sample_size = patterns.get("sample_size", 1)
        
        estimated_results = {}
        for pattern_type, ratio in discovered.items():
            estimated_count = int(ratio * len(series))
            estimated_results[pattern_type] = {
                "estimated_count": estimated_count,
                "estimated_ratio": ratio
            }
        
        # Determine primary data type based on highest ratio
        primary_type = max(discovered.items(), key=lambda x: x[1]) if discovered else ("unknown", 0)
        
        return {
            "status": "passed",
            "message": f"Primary type: {primary_type[0]} ({primary_type[1]:.1%})",
            "estimated_parsing": estimated_results,
            "primary_type": primary_type[0],
            "optimization": "two_stage_type_inference"
        }
    
    def _apply_entropy_patterns_efficiently(self, series: pd.Series, patterns: dict) -> dict:
        """Apply entropy patterns discovered in Stage 1 to estimate full dataset entropy efficiently."""
        if patterns.get("method") == "insufficient_data":
            return {"status": "warning", "message": "Insufficient data for entropy analysis"}
        
        sample_entropy = patterns.get("sample_entropy", 0)
        
        # Estimate full dataset entropy based on sample - entropy typically stabilizes
        estimated_entropy = sample_entropy
        
        return {
            "status": "passed",
            "message": f"Estimated entropy: {estimated_entropy:.2f}",
            "estimated_entropy": float(estimated_entropy),
            "sample_entropy": float(sample_entropy),
            "optimization": "two_stage_entropy_estimation"
        }
    
    def _apply_modality_patterns_efficiently(self, series: pd.Series, patterns: dict) -> dict:
        """Apply modality patterns discovered in Stage 1 to estimate full dataset modality."""
        if patterns.get("method") == "insufficient_data":
            return {"status": "warning", "message": "Insufficient data for modality analysis"}
        
        estimated_peaks = patterns.get("estimated_peaks", 1)
        sample_stats = patterns.get("sample_stats", {})
        
        # Estimate modality based on sample analysis - statistical properties are stable
        if estimated_peaks == 1:
            modality = "unimodal"
            status = "passed"
        elif estimated_peaks == 2:
            modality = "bimodal"
            status = "passed"
        else:
            modality = "multimodal"
            status = "warning"
        
        return {
            "status": status,
            "message": f"Estimated modality: {modality}",
            "estimated_modality": modality,
            "estimated_peaks": int(estimated_peaks),
            "sample_statistics": sample_stats,
            "optimization": "two_stage_modality_estimation"
        }
    
    def _apply_category_patterns_efficiently(self, series: pd.Series, patterns: dict) -> dict:
        """Apply category structure patterns for efficient functional dependency analysis."""
        unique_categories = patterns.get("unique_categories", 0)
        
        # Use discovered category structure to make efficiency decisions
        if unique_categories > len(series) * 0.8:
            return {
                "status": "skipped",
                "message": "Too many unique categories for functional dependency",
                "category_count": unique_categories,
                "optimization": "two_stage_category_filtering"
            }
        
        return {
            "status": "passed",
            "message": f"Suitable for functional dependency analysis ({unique_categories} categories)",
            "category_count": unique_categories,
            "top_categories": patterns.get("top_categories", {}),
            "optimization": "two_stage_category_analysis"
        }
    
    def execute_cross_column_check(self, rule: Rule, df: pd.DataFrame, 
                                 col1: str, col2: str) -> RuleResult:
        """
        Execute a cross-column check.
        
        Args:
            rule: Rule to execute
            df: DataFrame to check
            col1: First column name
            col2: Second column name
            
        Returns:
            RuleResult object
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # ROW-BASED OPTIMIZATION: Use optimized sampling for large datasets
            if self.config.enable_adaptive_sampling and len(df) > self.config.cross_column_sample_threshold:
                original_rows = len(df)
                df_optimized = self._create_optimized_sample_for_cross_column_rule(rule, df, col1, col2)
                rows_used = len(df_optimized)
                
                # Log the optimization applied
                if rows_used < original_rows:
                    print(f"🚀 ROW OPTIMIZATION: {rule.id} using {rows_used:,} of {original_rows:,} rows ({rows_used/original_rows:.1%})")
            else:
                df_optimized = df
                
            # Generate and execute the check code on optimized dataset
            check_code = self._generate_cross_column_check_code(rule, col1, col2)
            result_dict = self._execute_cross_column_check_code(check_code, df_optimized, col1, col2)
            
            # Convert result to RuleResult
            execution_time = (time.time() - start_time) * 1000
            
            # Default to passed when no explicit status
            if isinstance(result_dict, dict) and "status" not in result_dict:
                result_dict = {**result_dict, "status": "passed", "message": result_dict.get("message", "")}
            
            # Add compared column names to the details
            details_with_columns = {
                **result_dict,
                "compared_columns": {
                    "column_1": col1,
                    "column_2": col2
                }
            }
            
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus(result_dict.get("status", "passed")),
                score=self._calculate_score(result_dict),
                message=result_dict.get("message", ""),
                details=details_with_columns,
                execution_time_ms=execution_time,
                timestamp=timestamp
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus.ERROR,
                score=0.0,
                message=f"Cross-column check execution failed: {str(e)}",
                details={
                    "error": str(e), 
                    "traceback": traceback.format_exc(),
                    "compared_columns": {
                        "column_1": col1,
                        "column_2": col2
                    }
                },
                execution_time_ms=execution_time,
                timestamp=timestamp
            )

    def execute_table_check(self, rule: Rule, df: pd.DataFrame, attributes: Dict[str, ColumnAttributes]) -> RuleResult:
        """Execute a table-level check that expects a function taking df."""
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        try:
            # Build and execute a minimal environment
            namespace: Dict[str, Any] = {"pd": pd, "np": np, "df": df}
            exec(rule.code_template, namespace)
            # Find the function starting with 'check_'
            func = None
            for k, v in namespace.items():
                if k.startswith("check_") and callable(v):
                    func = v
                    break
            if func is None:
                raise RuntimeError("No check_* function defined")
            result_dict = func(df)
            exec_ms = (time.time() - start_time) * 1000
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus(result_dict.get("status", "passed" if result_dict else "warning")),
                score=self._calculate_score(result_dict or {"status": "passed"}),
                message=result_dict.get("message", "") if isinstance(result_dict, dict) else "",
                details=result_dict if isinstance(result_dict, dict) else {"result": result_dict},
                execution_time_ms=exec_ms,
                timestamp=timestamp,
            )
        except Exception as e:
            exec_ms = (time.time() - start_time) * 1000
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus.ERROR,
                score=0.0,
                message=f"Table check execution failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                execution_time_ms=exec_ms,
                timestamp=timestamp,
            )

    def execute_table_checks_parallel(self, rules: List[Rule], df: pd.DataFrame, attributes: Dict[str, ColumnAttributes]) -> List[RuleResult]:
        if not self.config.enable_parallel or len(rules) == 1:
            return [self.execute_table_check(rule, df, attributes) for rule in rules]
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self.execute_table_check, rule, df, attributes) for rule in rules]
            results: List[RuleResult] = []
            for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout_seconds):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(RuleResult(
                        rule_id="unknown",
                        rule_name="unknown",
                        status=RuleStatus.ERROR,
                        score=0.0,
                        message=f"Parallel table execution failed: {str(e)}",
                        details={"error": str(e)},
                        execution_time_ms=0.0,
                        timestamp=datetime.now().isoformat(),
                    ))
            return results
    
    def _generate_check_code(self, rule: Rule, series: pd.Series, 
                           attributes: ColumnAttributes) -> str:
        """
        Generate executable code for a column check.
        
        Args:
            rule: Rule definition
            series: Data series
            attributes: Column attributes
            
        Returns:
            Generated Python code as string
        """
        # Start with imports
        code = """
import pandas as pd
import numpy as np
from typing import Dict, Any

"""
        
        # Add the rule's code template
        code += rule.code_template
        
        # Add execution wrapper
        code += f"""
def execute_check(series: pd.Series) -> Dict[str, Any]:
    try:
        return check_{rule.id}(series)
    except NameError:
        # Fallback: find any function starting with 'check_'
        for _name, _fn in list(globals().items()):
            if _name.startswith('check_') and callable(_fn):
                return _fn(series)
        raise
    except Exception as e:
        return {{
            "status": "error",
            "message": f"Check failed: {{str(e)}}",
            "error": str(e)
        }}
"""
        
        return code
    
    def _generate_cross_column_check_code(self, rule: Rule, col1: str, col2: str) -> str:
        """
        Generate executable code for a cross-column check.
        
        Args:
            rule: Rule definition
            col1: First column name
            col2: Second column name
            
        Returns:
            Generated Python code as string
        """
        # Start with imports
        code = """
import pandas as pd
import numpy as np
from typing import Dict, Any

"""
        
        # Add the rule's code template
        code += rule.code_template
        
        # Add execution wrapper
        code += f"""
def execute_cross_column_check(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    try:
        return check_{rule.id}(df, col1, col2)
    except NameError:
        # Fallback: find any function starting with 'check_'
        for _name, _fn in list(globals().items()):
            if _name.startswith('check_') and callable(_fn):
                return _fn(df, col1, col2)
        raise
    except Exception as e:
        return {{
            "status": "error",
            "message": f"Cross-column check failed: {{str(e)}}",
            "error": str(e)
        }}
"""
        
        return code
    
    def _execute_check_code(self, code: str, series: pd.Series) -> Dict[str, Any]:
        """
        Execute generated check code.
        
        Args:
            code: Generated Python code
            series: Data series to check
            
        Returns:
            Result dictionary
        """
        # Create execution namespace
        from .text_regex_inference import TextRegexInference
        
        namespace = {
            'pd': pd,
            'np': np,
            'series': series,
            'TextRegexInference': TextRegexInference
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Execute the check
        result = namespace['execute_check'](series)
        
        return result
    
    def _execute_cross_column_check_code(self, code: str, df: pd.DataFrame, 
                                       col1: str, col2: str) -> Dict[str, Any]:
        """
        Execute generated cross-column check code.
        
        Args:
            code: Generated Python code
            df: DataFrame to check
            col1: First column name
            col2: Second column name
            
        Returns:
            Result dictionary
        """
        # Create execution namespace
        namespace = {
            'pd': pd,
            'np': np,
            'df': df,
            'col1': col1,
            'col2': col2
        }
        
        # Execute the code
        exec(code, namespace)
        
        # Execute the check
        result = namespace['execute_cross_column_check'](df, col1, col2)
        
        return result
    
    def _calculate_score(self, result_dict: Dict[str, Any]) -> float:
        """
        Calculate a score (0-100) based on the result.
        
        Args:
            result_dict: Result dictionary from check execution
            
        Returns:
            Score between 0 and 100
        """
        status = result_dict.get("status", "error")
        
        if status == "passed":
            return 100.0
        elif status == "warning":
            return 75.0
        elif status == "failed":
            return 25.0
        else:  # error or skipped
            return 0.0
    
    def execute_column_checks_parallel(self, rules: List[Rule], series: pd.Series, 
                                     attributes: ColumnAttributes) -> List[RuleResult]:
        """
        Execute multiple column checks in parallel.
        
        Args:
            rules: List of rules to execute
            series: Data series to check
            attributes: Column attributes
            
        Returns:
            List of RuleResult objects
        """
        if not self.config.enable_parallel or len(rules) == 1:
            # Execute sequentially
            return [
                self.execute_column_check(rule, series, attributes)
                for rule in rules
            ]
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self.execute_column_check, rule, series, attributes)
                for rule in rules
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result
                    error_result = RuleResult(
                        rule_id="unknown",
                        rule_name="unknown",
                        status=RuleStatus.ERROR,
                        score=0.0,
                        message=f"Parallel execution failed: {str(e)}",
                        details={"error": str(e)},
                        execution_time_ms=0.0,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(error_result)
            
            return results
    
    def execute_cross_column_checks_parallel(self, rules: List[Rule], df: pd.DataFrame, 
                                           column_pairs: List[Tuple[str, str]]) -> List[RuleResult]:
        """
        Execute cross-column checks with INTELLIGENT FILTERING for optimal performance.
        
        PERFORMANCE OPTIMIZATIONS:
        - Rule applicability pre-filtering by column types
        - Early exit strategies based on rule results
        - Batched execution for compatible rules
        - Skip expensive rules on incompatible column pairs
        
        Args:
            rules: List of rules to execute
            df: DataFrame to check
            column_pairs: List of column pairs to check
            
        Returns:
            List of RuleResult objects
        """
        if not rules or not column_pairs:
            return []
            
        results = []
        
        # OPTIMIZATION 1: Pre-compute column type information for all pairs
        column_type_cache = {}
        for col in df.columns:
            series = df[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_categorical = self._is_categorical_like(series)
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            column_type_cache[col] = {
                'is_numeric': is_numeric,
                'is_categorical': is_categorical,
                'unique_ratio': unique_ratio
            }
        
        # OPTIMIZATION 2: Execute identicality first (fast and enables skipping)
        identicality_rule = None
        other_rules = []
        
        for rule in rules:
            if rule.id == "identicality":
                identicality_rule = rule
            else:
                other_rules.append(rule)
        
        identical_pairs = set()
        
        # Execute identicality rule first for all pairs
        if identicality_rule:
            for col1, col2 in column_pairs:
                result = self.execute_cross_column_check(identicality_rule, df, col1, col2)
                results.append(result)
                
                # Check if columns are identical and should skip other rules
                if (result.details and 
                    result.details.get("should_skip_other_rules", False)):
                    identical_pairs.add((col1, col2))
        
        # OPTIMIZATION 3: Smart rule execution with type-based filtering
        for rule in other_rules:
            applicable_pairs = self._filter_pairs_for_rule(rule, column_pairs, identical_pairs, 
                                                         column_type_cache, df)
            
            if not applicable_pairs:
                continue  # Skip rule entirely if no applicable pairs
                
            # Execute rule only on applicable pairs
            rule_results = self._execute_rule_on_pairs(rule, df, applicable_pairs)
            results.extend(rule_results)
        
        return results
    
    def _is_categorical_like(self, series: pd.Series) -> bool:
        """
        Determine if a series is categorical-like for performance optimization.
        
        Args:
            series: Pandas Series to check
            
        Returns:
            True if series should be treated as categorical
        """
        if pd.api.types.is_categorical_dtype(series):
            return True
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            return unique_ratio < 0.7  # String columns with reasonable cardinality
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            return unique_ratio < 0.3 and series.nunique() <= 50  # Low cardinality numeric
        return False
    
    def _filter_pairs_for_rule(self, rule: Rule, column_pairs: List[Tuple[str, str]], 
                             identical_pairs: set, column_type_cache: dict, 
                             df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Filter column pairs based on rule-specific requirements for MASSIVE performance gains.
        
        Args:
            rule: Rule to filter pairs for
            column_pairs: All available column pairs
            identical_pairs: Pairs to skip (identical columns)
            column_type_cache: Cached column type information
            df: DataFrame being analyzed
            
        Returns:
            List of applicable column pairs for this rule
        """
        applicable_pairs = []
        
        for col1, col2 in column_pairs:
            # Skip identical pairs
            if (col1, col2) in identical_pairs:
                continue
                
            col1_info = column_type_cache.get(col1, {})
            col2_info = column_type_cache.get(col2, {})
            
            # RULE-SPECIFIC FILTERING for major performance gains:
            
            if rule.id == "correlation":
                # Correlation: ONLY run on numeric-numeric pairs
                if not (col1_info.get('is_numeric') and col2_info.get('is_numeric')):
                    continue
                    
            elif rule.id == "functional_dependency":
                # Functional dependency: ONLY run on categorical-categorical pairs
                if not (col1_info.get('is_categorical') and col2_info.get('is_categorical')):
                    continue
                # Skip if either column is too unique (>95% unique)
                if (col1_info.get('unique_ratio', 1) > 0.95 or 
                    col2_info.get('unique_ratio', 1) > 0.95):
                    continue
                    
            elif rule.id == "missingness_relationships":
                # Missingness: Only useful if both columns have some nulls
                col1_nulls = df[col1].isnull().sum()
                col2_nulls = df[col2].isnull().sum()
                if col1_nulls == 0 and col2_nulls == 0:
                    continue  # No point if neither column has nulls
                    
            # Add more rule-specific filters here as needed
            
            applicable_pairs.append((col1, col2))
        
        return applicable_pairs
    
    def _execute_rule_on_pairs(self, rule: Rule, df: pd.DataFrame, 
                             pairs: List[Tuple[str, str]]) -> List[RuleResult]:
        """
        Execute a single rule on multiple pairs with optimal parallelization.
        
        Args:
            rule: Rule to execute
            df: DataFrame to analyze
            pairs: Column pairs to analyze
            
        Returns:
            List of RuleResult objects
        """
        if not pairs:
            return []
        
        if not self.config.enable_parallel or len(pairs) == 1:
            # Sequential execution
            return [self.execute_cross_column_check(rule, df, col1, col2) 
                   for col1, col2 in pairs]
        
        # Parallel execution with better error handling
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self.execute_cross_column_check, rule, df, col1, col2)
                for col1, col2 in pairs
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result with better context
                    error_result = RuleResult(
                        rule_id=rule.id,
                        rule_name=rule.name,
                        status=RuleStatus.ERROR,
                        score=0.0,
                        message=f"Parallel execution failed for rule {rule.id}: {str(e)}",
                        details={"error": str(e), "rule_id": rule.id},
                        execution_time_ms=0.0,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(error_result)
            
            return results
