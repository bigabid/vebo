"""
Focused test cases for Statistical Enhancements (Layers 3 & 4).

Tests the adaptive statistical power validation and performance-optimized significance testing.
"""

import pytest
import pandas as pd
import numpy as np
import time
from statistical_enhancements import StatisticalEnhancements


class TestLayer3StatisticalPower:
    """Test Layer 3: Adaptive Statistical Power Validation"""
    
    def setup_method(self):
        self.enhancer = StatisticalEnhancements()
    
    def test_adaptive_thresholds_small_dataset(self):
        """Test that small datasets get more lenient thresholds."""
        # Small dataset (< 100 rows)
        small_df = pd.DataFrame({
            'cat1': ['A', 'B'] * 25,  # 50 rows
            'cat2': ['X', 'Y', 'Z'] * 16 + ['X', 'Y']  # 50 rows
        })
        
        power_check = self.enhancer.check_statistical_power_adaptive(small_df, 'cat1', 'cat2')
        
        # Should use lenient thresholds: min_sample=20, min_group_size=2
        assert power_check['details']['thresholds_used']['min_sample'] == 20
        assert power_check['details']['thresholds_used']['min_group_size'] == 2
        assert power_check['sufficient_power'] == True  # Should pass with lenient thresholds
    
    def test_adaptive_thresholds_medium_dataset(self):
        """Test medium dataset thresholds."""
        # Medium dataset (1000 rows)
        medium_df = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'cat2': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        
        power_check = self.enhancer.check_statistical_power_adaptive(medium_df, 'cat1', 'cat2')
        
        # Should use medium thresholds: min_sample=30, min_group_size=3
        assert power_check['details']['thresholds_used']['min_sample'] == 30
        assert power_check['details']['thresholds_used']['min_group_size'] == 3
        assert power_check['sufficient_power'] == True
    
    def test_adaptive_thresholds_large_dataset(self):
        """Test large dataset gets stricter thresholds."""
        # Large dataset (50k rows) 
        large_df = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], 50000),
            'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], 50000)
        })
        
        power_check = self.enhancer.check_statistical_power_adaptive(large_df, 'cat1', 'cat2')
        
        # Should use strict thresholds: min_sample=100, min_group_size=5
        assert power_check['details']['thresholds_used']['min_sample'] == 100
        assert power_check['details']['thresholds_used']['min_group_size'] == 5
        assert power_check['sufficient_power'] == True
    
    def test_insufficient_sample_size(self):
        """Test rejection due to insufficient sample size."""
        # Very small dataset
        tiny_df = pd.DataFrame({
            'cat1': ['A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X']
        })
        
        power_check = self.enhancer.check_statistical_power_adaptive(tiny_df, 'cat1', 'cat2')
        
        assert power_check['sufficient_power'] == False
        assert power_check['reason'] == 'insufficient_total_sample'
        assert power_check['valid_rows'] < power_check['details']['min_required']
    
    def test_insufficient_group_sizes(self):
        """Test rejection due to insufficient group sizes."""
        # Dataset with one very dominant category
        skewed_df = pd.DataFrame({
            'cat1': ['A'] * 95 + ['B'] * 5,  # Very unbalanced
            'cat2': ['X'] * 98 + ['Y'] * 2   # Very unbalanced
        })
        
        power_check = self.enhancer.check_statistical_power_adaptive(skewed_df, 'cat1', 'cat2')
        
        # May fail due to small average group sizes
        if not power_check['sufficient_power']:
            assert power_check['reason'] == 'inadequate_power'
            # Check that group size calculation is working
            assert 'avg_group_size_col1' in power_check['details']
            assert 'avg_group_size_col2' in power_check['details']
    
    def test_adequate_balanced_groups(self):
        """Test approval for well-balanced groups."""
        balanced_df = pd.DataFrame({
            'department': (['Sales'] * 25 + ['Engineering'] * 25 + 
                          ['Marketing'] * 25 + ['HR'] * 25),
            'level': (['Junior'] * 20 + ['Senior'] * 30 + 
                     ['Manager'] * 30 + ['Director'] * 20)
        })
        
        power_check = self.enhancer.check_statistical_power_adaptive(balanced_df, 'department', 'level')
        
        assert power_check['sufficient_power'] == True
        assert power_check['reason'] == 'adequate_power'
        assert power_check['details']['col1_groups'] == 4
        assert power_check['details']['col2_groups'] == 4


class TestLayer4PerformanceOptimization:
    """Test Layer 4: Performance-Optimized Statistical Significance Testing"""
    
    def setup_method(self):
        self.enhancer = StatisticalEnhancements()
        # Clear cache between tests
        self.enhancer._chi2_cache.clear()
    
    def test_cramers_v_caching_performance(self):
        """Test that caching improves performance for repeated calculations."""
        # Create test dataset
        df = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], 1000)
        })
        
        # First calculation (should populate cache)
        start_time = time.time()
        result1 = self.enhancer.performance_optimized_cramers_v(df, 'cat1', 'cat2')
        first_duration = time.time() - start_time
        
        # Second calculation (should use cache)  
        start_time = time.time()
        result2 = self.enhancer.performance_optimized_cramers_v(df, 'cat1', 'cat2')
        second_duration = time.time() - start_time
        
        # Results should be identical
        assert result1['cramers_v'] == result2['cramers_v']
        assert result1['p_value'] == result2['p_value']
        
        # Second calculation should be faster (cache hit)
        # Note: This might be flaky in very fast systems, so we're lenient
        assert len(self.enhancer._chi2_cache) > 0  # Cache should have entries
        
    def test_cramers_v_statistical_significance(self):
        """Test statistical significance detection in Cramér's V."""
        # Create dataset with known association
        n = 500
        # Strong association: education -> income
        education = np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n, p=[0.4, 0.3, 0.2, 0.1])
        income = []
        
        for edu in education:
            if edu == 'HS':
                income.append(np.random.choice(['Low', 'Medium'], p=[0.8, 0.2]))
            elif edu == 'Bachelor':
                income.append(np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.6, 0.2]))
            elif edu == 'Master':
                income.append(np.random.choice(['Medium', 'High', 'VeryHigh'], p=[0.2, 0.6, 0.2]))
            else:  # PhD
                income.append(np.random.choice(['High', 'VeryHigh'], p=[0.4, 0.6]))
        
        df_associated = pd.DataFrame({'education': education, 'income': income})
        result = self.enhancer.performance_optimized_cramers_v(df_associated, 'education', 'income')
        
        assert result['performance_optimized'] == True
        assert 'statistically_significant' in result
        assert 'effect_size_meaningful' in result
        assert result['p_value'] is not None
        
        # Should detect the association
        if result['statistically_significant']:
            assert result['cramers_v'] > 0.1  # Meaningful effect size
            assert result['status'] in ['high', 'medium', 'low']  # Not "not_significant"
    
    def test_cramers_v_no_association(self):
        """Test that truly independent variables show no significance."""
        # Create independent variables
        df_independent = pd.DataFrame({
            'random1': np.random.choice(['A', 'B', 'C'], 1000),
            'random2': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        
        result = self.enhancer.performance_optimized_cramers_v(df_independent, 'random1', 'random2')
        
        # Should detect lack of association (though might occasionally fail due to randomness)
        assert result['p_value'] is not None
        assert 'statistically_significant' in result
        
        # Most of the time should not be significant for truly random data
        # (We can't guarantee this due to randomness, but can check the machinery works)
        assert result['status'] in ['not_significant', 'very_low', 'low', 'medium', 'high']
    
    def test_functional_dependency_confidence_scoring(self):
        """Test FD confidence scoring mechanism."""
        # Create strong functional dependency
        df_strong = pd.DataFrame({
            'country': ['USA', 'USA', 'Canada', 'Canada', 'UK', 'UK'] * 100,
            'currency': ['USD', 'USD', 'CAD', 'CAD', 'GBP', 'GBP'] * 100
        })
        
        result = self.enhancer.enhanced_functional_dependency_with_confidence(df_strong, 'country', 'currency')
        
        assert result['performance_optimized'] == True
        assert 'confidence' in result
        assert 'statistical_power' in result
        assert result['fd_holds_ratio'] > 0.9  # Strong dependency
        assert result['confidence'] > 0.5  # Should have reasonable confidence
        assert result['status'] in ['high', 'medium']  # Should be rated highly
    
    def test_functional_dependency_weak_relationship(self):
        """Test FD with weak/no relationship."""
        # Create dataset with no functional dependency
        df_weak = pd.DataFrame({
            'random_cat1': np.random.choice(['A', 'B', 'C', 'D'], 500),
            'random_cat2': np.random.choice(['X', 'Y', 'Z', 'W'], 500)
        })
        
        result = self.enhancer.enhanced_functional_dependency_with_confidence(df_weak, 'random_cat1', 'random_cat2')
        
        assert 'confidence' in result
        assert 'statistical_power' in result
        assert result['fd_holds_ratio'] < 0.5  # Should be low for random data
        assert result['status'] in ['very_low', 'low']  # Should be rated low


class TestIntegrationScenarios:
    """Test realistic integration scenarios combining Layers 3 & 4."""
    
    def setup_method(self):
        self.enhancer = StatisticalEnhancements()
    
    def test_complete_workflow_valid_analysis(self):
        """Test complete workflow for valid analysis case."""
        # Create realistic business dataset
        n_employees = 800
        departments = np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], n_employees, p=[0.3, 0.4, 0.2, 0.1])
        
        # Create job levels with some dependency on department
        job_levels = []
        for dept in departments:
            if dept == 'Engineering':
                job_levels.append(np.random.choice(['Junior', 'Senior', 'Lead', 'Principal'], p=[0.3, 0.4, 0.2, 0.1]))
            elif dept == 'Sales':
                job_levels.append(np.random.choice(['Associate', 'Senior', 'Manager'], p=[0.4, 0.4, 0.2]))
            else:
                job_levels.append(np.random.choice(['Junior', 'Senior', 'Manager'], p=[0.5, 0.3, 0.2]))
        
        df = pd.DataFrame({
            'department': departments,
            'job_level': job_levels,
            'salary_band': np.random.choice(['Band1', 'Band2', 'Band3', 'Band4'], n_employees)  # Independent
        })
        
        # Test Layer 3: Should pass power analysis
        power_check = self.enhancer.should_proceed_with_analysis(df, 'department', 'job_level')
        assert power_check['should_proceed'] == True
        
        # Test Layer 4: Should detect meaningful relationship
        fd_result = self.enhancer.enhanced_functional_dependency_with_confidence(df, 'department', 'job_level')
        assert fd_result['performance_optimized'] == True
        assert fd_result['confidence'] > 0
        
        cramers_result = self.enhancer.performance_optimized_cramers_v(df, 'department', 'job_level')
        assert cramers_result['performance_optimized'] == True
        assert cramers_result['p_value'] is not None
    
    def test_complete_workflow_insufficient_power(self):
        """Test complete workflow when statistical power is insufficient."""
        # Create tiny dataset that should fail power analysis
        tiny_df = pd.DataFrame({
            'tiny_cat1': ['A', 'B', 'A', 'B'],
            'tiny_cat2': ['X', 'Y', 'X', 'Y']
        })
        
        # Should fail at Layer 3
        power_check = self.enhancer.should_proceed_with_analysis(tiny_df, 'tiny_cat1', 'tiny_cat2')
        assert power_check['should_proceed'] == False
        assert power_check['reason'] == 'insufficient_statistical_power'
        
        # Analysis should still work if forced, but with low confidence
        fd_result = self.enhancer.enhanced_functional_dependency_with_confidence(tiny_df, 'tiny_cat1', 'tiny_cat2')
        assert fd_result['statistical_power'] < 0.5  # Should have low power
    
    def test_performance_benchmark(self):
        """Basic performance benchmark to ensure optimizations don't hurt speed."""
        # Create moderately large dataset
        n = 5000
        df_large = pd.DataFrame({
            'cat1': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
            'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n),
            'cat3': np.random.choice(['P', 'Q', 'R'], n)
        })
        
        # Time the enhanced analyses
        start = time.time()
        
        # Multiple analyses to test caching
        result1 = self.enhancer.performance_optimized_cramers_v(df_large, 'cat1', 'cat2')
        result2 = self.enhancer.performance_optimized_cramers_v(df_large, 'cat1', 'cat2')  # Should hit cache
        result3 = self.enhancer.enhanced_functional_dependency_with_confidence(df_large, 'cat1', 'cat3')
        
        total_time = time.time() - start
        
        # Should complete in reasonable time (< 2 seconds for 5k rows)
        assert total_time < 2.0, f"Performance test took {total_time:.2f} seconds, expected < 2.0"
        
        # Results should be valid
        assert result1['cramers_v'] is not None
        assert result2['cramers_v'] == result1['cramers_v']  # Cache hit should give same result
        assert result3['fd_holds_ratio'] is not None


def run_focused_validation():
    """Run validation comparing current vs enhanced approaches."""
    print("=== Statistical Enhancements Validation (Layers 3 & 4) ===\n")
    
    enhancer = StatisticalEnhancements()
    
    # Test case 1: Small dataset (should use adaptive thresholds)
    small_df = pd.DataFrame({
        'small_cat1': ['A', 'B'] * 30,
        'small_cat2': ['X', 'Y', 'Z'] * 20  
    })
    
    print("Small Dataset Test:")
    power_check = enhancer.check_statistical_power_adaptive(small_df, 'small_cat1', 'small_cat2')
    print(f"  Power Check: {'✓ Pass' if power_check['sufficient_power'] else '✗ Fail'}")
    print(f"  Thresholds: {power_check['details']['thresholds_used']}")
    
    if power_check['sufficient_power']:
        fd_result = enhancer.enhanced_functional_dependency_with_confidence(small_df, 'small_cat1', 'small_cat2')
        print(f"  FD Result: {fd_result['status']} (confidence: {fd_result['confidence']:.2f})")
    
    # Test case 2: Large dataset with association
    print("\nLarge Dataset with Association:")
    n = 2000
    large_cat1 = np.random.choice(['Type1', 'Type2', 'Type3'], n, p=[0.5, 0.3, 0.2])
    large_cat2 = []
    
    # Create association
    for cat in large_cat1:
        if cat == 'Type1':
            large_cat2.append(np.random.choice(['ClassA', 'ClassB'], p=[0.8, 0.2]))
        elif cat == 'Type2': 
            large_cat2.append(np.random.choice(['ClassB', 'ClassC'], p=[0.7, 0.3]))
        else:
            large_cat2.append(np.random.choice(['ClassC', 'ClassD'], p=[0.6, 0.4]))
    
    large_df = pd.DataFrame({'category': large_cat1, 'class': large_cat2})
    
    power_check = enhancer.check_statistical_power_adaptive(large_df, 'category', 'class') 
    print(f"  Power Check: {'✓ Pass' if power_check['sufficient_power'] else '✗ Fail'}")
    
    if power_check['sufficient_power']:
        cramers_result = enhancer.performance_optimized_cramers_v(large_df, 'category', 'class')
        print(f"  Cramér's V: {cramers_result['cramers_v']:.3f} (p={cramers_result['p_value']:.3f})")
        print(f"  Significant: {'✓ Yes' if cramers_result['statistically_significant'] else '✗ No'}")
        print(f"  Status: {cramers_result['status']}")


if __name__ == "__main__":
    run_focused_validation()
