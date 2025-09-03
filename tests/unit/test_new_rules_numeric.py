"""
Unit tests for new numeric rules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.rule_engine import RuleEngine
from vebo_profiler.core.meta_rules import ColumnAttributes
from vebo_profiler.core.check_executor import CheckExecutor


def _attrs(name: str, series: pd.Series) -> ColumnAttributes:
    engine = RuleEngine()
    return engine.meta_detector.analyze_column(series.rename(name))


class TestNewNumericRules:
    def test_outlier_detection_zscore(self):
        engine = RuleEngine()
        s = pd.Series([1,2,2,3,3,3,100])
        attrs = _attrs('num', s)
        rule = engine.get_rule('outlier_detection_zscore')
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert 'outlier_count' in res.details

    def test_numeric_histogram_quantiles(self):
        engine = RuleEngine()
        s = pd.Series(np.random.randn(200))
        attrs = _attrs('num', s)
        rule = engine.get_rule('numeric_histogram_quantiles')
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert 'histogram' in res.details
        assert 'quantiles' in res.details

    def test_modality_estimation(self):
        engine = RuleEngine()
        s = pd.Series(list(np.random.normal(-2, 0.5, 200)) + list(np.random.normal(2, 0.5, 200)))
        attrs = _attrs('num', s)
        rule = engine.get_rule('modality_estimation')
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert 'modality' in res.details

