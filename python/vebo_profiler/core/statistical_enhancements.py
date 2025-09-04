"""
Statistical Enhancements for Cross-Column Analysis
Focused implementation of Layers 3 & 4 only:
- Layer 3: Statistical Power Validation (data-size adaptive)  
- Layer 4: Performance-Optimized Statistical Significance Testing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from scipy.stats import chi2_contingency
import warnings


class StatisticalEnhancements:
    """Statistical enhancements for cross-column analysis without restrictive filtering."""
    
    def __init__(self):
        # Layer 3: Adaptive thresholds based on data size
        self.base_min_sample_size = 30
        self.base_min_group_size = 3
        
        # Layer 4: Performance-optimized significance testing
        self.significance_level = 0.05
        self.min_effect_size_cramers = 0.1
        
        # Performance optimization: cache chi2 calculations
        self._chi2_cache = {}
    
    def check_statistical_power_adaptive(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Layer 3: Adaptive statistical power validation based on data size.
        
        Adjusts thresholds based on dataset size to ensure reliable analysis.
        """
        n_rows = len(df)
        
        # Adaptive thresholds based on data size
        if n_rows < 100:
            min_sample = 20
            min_group_size = 2
        elif n_rows < 1000:
            min_sample = 30
            min_group_size = 3
        elif n_rows < 10000:
            min_sample = 50
            min_group_size = 4
        else:
            min_sample = 100
            min_group_size = 5
        
        # Check overall sample size
        valid_rows = len(df[[col1, col2]].dropna())
        if valid_rows < min_sample:
            return {
                'sufficient_power': False,
                'reason': 'insufficient_total_sample',
                'valid_rows': valid_rows,
                'required_min': min_sample,
                'details': {
                    'total_rows': n_rows,
                    'valid_pairs': valid_rows,
                    'min_required': min_sample
                }
            }
        
        # Check group sizes for meaningful analysis
        col1_groups = df[col1].nunique()
        col2_groups = df[col2].nunique()
        
        avg_group_size_col1 = valid_rows / col1_groups if col1_groups > 0 else 0
        avg_group_size_col2 = valid_rows / col2_groups if col2_groups > 0 else 0
        
        sufficient_group_sizes = (
            avg_group_size_col1 >= min_group_size and
            avg_group_size_col2 >= min_group_size
        )
        
        # Additional check: ensure we have enough groups
        min_groups = 2 if n_rows < 100 else 3
        sufficient_groups = col1_groups >= min_groups and col2_groups >= min_groups
        
        sufficient_power = sufficient_group_sizes and sufficient_groups
        
        return {
            'sufficient_power': sufficient_power,
            'reason': 'adequate_power' if sufficient_power else 'inadequate_power',
            'details': {
                'total_rows': n_rows,
                'valid_pairs': valid_rows,
                'col1_groups': col1_groups,
                'col2_groups': col2_groups,
                'avg_group_size_col1': avg_group_size_col1,
                'avg_group_size_col2': avg_group_size_col2,
                'min_group_size_required': min_group_size,
                'min_groups_required': min_groups,
                'thresholds_used': {
                    'min_sample': min_sample,
                    'min_group_size': min_group_size,
                    'min_groups': min_groups
                }
            }
        }
    
    def performance_optimized_cramers_v(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Layer 4: Performance-optimized Cramér's V with statistical significance.
        
        Uses caching and efficient computation to minimize performance impact.
        """
        try:
            # Create cache key for performance optimization
            col1_hash = hash(tuple(sorted(df[col1].dropna().astype(str))))
            col2_hash = hash(tuple(sorted(df[col2].dropna().astype(str))))
            cache_key = (col1_hash, col2_hash, len(df))
            
            # Check cache first (performance optimization)
            if cache_key in self._chi2_cache:
                cached_result = self._chi2_cache[cache_key]
                chi2, p_value, dof = cached_result['chi2'], cached_result['p_value'], cached_result['dof']
                ct_shape = cached_result['ct_shape']
                n = cached_result['n']
            else:
                # Efficient contingency table creation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ct = pd.crosstab(df[col1].astype(str), df[col2].astype(str))
                
                # Quick validity checks
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    return {
                        "status": "skipped", 
                        "message": "Insufficient categories for analysis", 
                        "cramers_v": None,
                        "reason": "insufficient_categories"
                    }
                
                n = ct.values.sum()
                if n == 0:
                    return {
                        "status": "skipped", 
                        "message": "No valid data pairs", 
                        "cramers_v": None,
                        "reason": "no_valid_pairs"
                    }
                
                # Performance-optimized chi-square test
                chi2, p_value, dof, expected = chi2_contingency(ct.values)
                
                # Cache the expensive computation
                self._chi2_cache[cache_key] = {
                    'chi2': chi2, 'p_value': p_value, 'dof': dof,
                    'ct_shape': ct.shape, 'n': n
                }
                ct_shape = ct.shape
            
            # Efficient Cramér's V calculation
            k = min(ct_shape)
            v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 and n > 0 else 0.0
            
            # Fast significance and effect size evaluation
            is_significant = p_value < self.significance_level
            has_meaningful_effect = v >= self.min_effect_size_cramers
            
            # Efficient status determination
            if not is_significant:
                status = "not_significant"
                strength = "not_significant"
            elif v >= 0.5 and p_value < 0.001:
                strength = "strong"
                status = "high"
            elif v >= 0.3 and p_value < 0.01:
                strength = "moderate"  
                status = "medium"
            elif has_meaningful_effect:
                strength = "weak"
                status = "low"
            else:
                strength = "very_weak"
                status = "very_low"
            
            return {
                "cramers_v": float(v),
                "p_value": float(p_value),
                "chi2_statistic": float(chi2),
                "degrees_of_freedom": int(dof),
                "strength": strength,
                "status": status,
                "message": f"Cramér's V: {v:.3f} (p={p_value:.3f}, {strength})",
                "statistically_significant": is_significant,
                "effect_size_meaningful": has_meaningful_effect,
                "enhanced_analysis": True,
                "performance_optimized": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in Cramér's V calculation: {str(e)}",
                "cramers_v": None,
                "reason": "computation_error"
            }
    
    def enhanced_functional_dependency_with_confidence(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """
        Layer 4: Enhanced functional dependency with statistical confidence measures.
        
        Adds confidence scoring without expensive statistical tests.
        """
        def check_direction_with_confidence(df, source_col, target_col):
            """Compute FD with confidence metrics."""
            # Efficient grouping operation
            grouped_data = df[[source_col, target_col]].dropna().groupby(source_col, dropna=False)[target_col]
            
            # Vectorized uniqueness check
            unique_counts = grouped_data.nunique()
            
            if len(unique_counts) == 0:
                return {
                    "fd_holds_ratio": 0.0,
                    "violating_groups": 0,
                    "total_groups": 0,
                    "confidence": 0.0,
                    "statistical_power": 0.0
                }
            
            # Efficient violation detection
            violating_groups = (unique_counts > 1).sum()
            total_groups = len(unique_counts)
            fd_holds_ratio = 1.0 - (violating_groups / total_groups)
            
            # Statistical confidence based on group distribution
            group_sizes = grouped_data.size()
            avg_group_size = group_sizes.mean()
            min_group_size = group_sizes.min()
            
            # Confidence scoring (0-1 scale)
            # Higher confidence for: more groups, larger average group size, more balanced groups
            size_confidence = min(1.0, avg_group_size / 5.0)  # Normalize around 5 obs per group
            balance_confidence = min(1.0, min_group_size / max(1, avg_group_size * 0.5))
            group_count_confidence = min(1.0, total_groups / 10.0)  # Normalize around 10 groups
            
            overall_confidence = (size_confidence + balance_confidence + group_count_confidence) / 3.0
            
            # Statistical power assessment (simple but effective)
            statistical_power = min(1.0, (total_groups * avg_group_size) / 100.0)
            
            return {
                "fd_holds_ratio": float(fd_holds_ratio),
                "violating_groups": int(violating_groups), 
                "total_groups": int(total_groups),
                "confidence": float(overall_confidence),
                "statistical_power": float(statistical_power),
                "group_stats": {
                    "avg_group_size": float(avg_group_size),
                    "min_group_size": int(min_group_size),
                    "max_group_size": int(group_sizes.max())
                }
            }
        
        # Analyze both directions efficiently
        fd_col1_to_col2 = check_direction_with_confidence(df, col1, col2)
        fd_col2_to_col1 = check_direction_with_confidence(df, col2, col1)
        
        # Determine best direction based on both strength and confidence
        score1 = fd_col1_to_col2["fd_holds_ratio"] * fd_col1_to_col2["confidence"]
        score2 = fd_col2_to_col1["fd_holds_ratio"] * fd_col2_to_col1["confidence"]
        
        if score1 >= score2:
            best_direction = f"{col1} -> {col2}"
            best_stats = fd_col1_to_col2
        else:
            best_direction = f"{col2} -> {col1}"
            best_stats = fd_col2_to_col1
        
        # Enhanced status determination with confidence consideration
        fd_ratio = best_stats["fd_holds_ratio"]
        confidence = best_stats["confidence"]
        power = best_stats["statistical_power"]
        
        # Multi-factor status assessment
        if fd_ratio >= 0.95 and confidence >= 0.7 and power >= 0.5:
            status = "high"
        elif fd_ratio >= 0.85 and confidence >= 0.5 and power >= 0.3:
            status = "medium"
        elif fd_ratio >= 0.75 and confidence >= 0.3:
            status = "low"
        else:
            status = "very_low"
        
        return {
            "best_direction": best_direction,
            "fd_holds_ratio": best_stats["fd_holds_ratio"],
            "confidence": best_stats["confidence"],
            "statistical_power": best_stats["statistical_power"],
            "violating_groups": best_stats["violating_groups"],
            "total_groups": best_stats["total_groups"],
            "status": status,
            "message": f"Enhanced FD: {best_direction} holds in {fd_ratio:.1%} (conf: {confidence:.2f}, power: {power:.2f})",
            "enhanced_analysis": True,
            "performance_optimized": True,
            "col1_to_col2_ratio": fd_col1_to_col2["fd_holds_ratio"],
            "col2_to_col1_ratio": fd_col2_to_col1["fd_holds_ratio"],
            "detailed_stats": {
                "col1_to_col2": fd_col1_to_col2,
                "col2_to_col1": fd_col2_to_col1
            }
        }
    
    def should_proceed_with_analysis(self, df: pd.DataFrame, col1: str, col2: str, 
                                   analysis_type: str = "both") -> Dict[str, Any]:
        """
        Combined Layer 3 check: Determine if analysis should proceed based on statistical power.
        
        Args:
            analysis_type: "fd", "cramers_v", or "both"
        """
        power_check = self.check_statistical_power_adaptive(df, col1, col2)
        
        if not power_check['sufficient_power']:
            return {
                'should_proceed': False,
                'reason': 'insufficient_statistical_power',
                'analysis_type': analysis_type,
                'power_details': power_check
            }
        
        return {
            'should_proceed': True,
            'reason': 'adequate_statistical_power',
            'analysis_type': analysis_type,
            'power_details': power_check
        }


# Integration helper functions for existing codebase
def create_enhanced_functional_dependency_template():
    """Generate the enhanced FD rule template for integration."""
    return '''
def check_functional_dependency(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    import pandas as pd
    from statistical_enhancements import StatisticalEnhancements
    
    enhancer = StatisticalEnhancements()
    
    # Layer 3: Check statistical power first
    power_check = enhancer.should_proceed_with_analysis(df, col1, col2, "fd")
    if not power_check['should_proceed']:
        return {
            "status": "skipped",
            "message": f"Skipped: {power_check['reason']}",
            "fd_holds_ratio": 0,
            "reason": power_check['reason'],
            "skip_details": power_check['power_details']
        }
    
    # Layer 4: Enhanced analysis with confidence measures
    return enhancer.enhanced_functional_dependency_with_confidence(df, col1, col2)
'''

def create_enhanced_cramers_v_template():
    """Generate the enhanced Cramér's V rule template for integration.""" 
    return '''
def check_categorical_association_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    import pandas as pd
    from statistical_enhancements import StatisticalEnhancements
    
    enhancer = StatisticalEnhancements()
    
    # Layer 3: Check statistical power first
    power_check = enhancer.should_proceed_with_analysis(df, col1, col2, "cramers_v")
    if not power_check['should_proceed']:
        return {
            "status": "skipped",
            "message": f"Skipped: {power_check['reason']}",
            "cramers_v": None,
            "reason": power_check['reason'],
            "skip_details": power_check['power_details']
        }
    
    # Layer 4: Performance-optimized Cramér's V with significance testing
    return enhancer.performance_optimized_cramers_v(df, col1, col2)
'''
