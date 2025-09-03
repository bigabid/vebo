"""
Unit tests for new basic stats rules added to RuleEngine.
"""

import pytest
import pandas as pd
import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.rule_engine import RuleEngine
from vebo_profiler.core.meta_rules import ColumnAttributes, TypeCategory, DiversityLevel, NullabilityLevel


def _attrs(name: str, series: pd.Series) -> ColumnAttributes:
    engine = RuleEngine()
    return engine.meta_detector.analyze_column(series.rename(name))


class TestNewBasicStatsRules:
    def test_top_k_frequencies(self):
        engine = RuleEngine()
        s = pd.Series([1, 1, 2, 3, 3, 3, None])
        attrs = _attrs('col', s)
        rule = engine.get_rule('top_k_frequencies')
        assert rule is not None
        result = engine.meta_detector  # just to satisfy lints for unused
        # Execute via CheckExecutor pathway
        from vebo_profiler.core.check_executor import CheckExecutor
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert res.status.value in ["passed", "warning"]
        assert 'top_k' in res.details
        assert 'other_ratio' in res.details

    def test_duplicate_value_analysis(self):
        engine = RuleEngine()
        s = pd.Series(['a', 'a', 'b', 'c'])
        attrs = _attrs('col', s)
        rule = engine.get_rule('duplicate_value_analysis')
        from vebo_profiler.core.check_executor import CheckExecutor
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert 'duplicate_count' in res.details
        assert res.details['duplicate_count'] == 1  # 4 total - 3 unique

    def test_parseability_analysis(self):
        engine = RuleEngine()
        s = pd.Series(['1', '2.5', 'x', '2020-01-01', '{"a":1}', None])
        attrs = _attrs('col', s)
        rule = engine.get_rule('parseability_analysis')
        from vebo_profiler.core.check_executor import CheckExecutor
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        for key in ['parseable_int_ratio','parseable_float_ratio','parseable_datetime_ratio','parseable_json_ratio']:
            assert key in res.details

    def test_stability_entropy(self):
        engine = RuleEngine()
        s = pd.Series(['a','a','a','b','b','c'])
        attrs = _attrs('col', s)
        rule = engine.get_rule('stability_entropy')
        from vebo_profiler.core.check_executor import CheckExecutor
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert 'entropy' in res.details
        assert 'normalized_entropy' in res.details

