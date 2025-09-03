"""
Unit tests for new textual rules.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.rule_engine import RuleEngine
from vebo_profiler.core.meta_rules import ColumnAttributes
from vebo_profiler.core.check_executor import CheckExecutor


def _attrs(name: str, series: pd.Series) -> ColumnAttributes:
    engine = RuleEngine()
    return engine.meta_detector.analyze_column(series.rename(name))


class TestNewTextualRules:
    def test_whitespace_encoding_checks(self):
        engine = RuleEngine()
        s = pd.Series([' a', 'b ', '   ', 'c\x07', 'ok'])
        attrs = _attrs('txt', s)
        rule = engine.get_rule('whitespace_encoding_checks')
        if rule is None:
            # Fallback: ensure textual rules loaded and fetch by id again
            rules = engine.get_rules_by_category('text_quality')
            rule = next((r for r in rules if r.id == 'whitespace_encoding_checks'), None)
        assert rule is not None
        execu = CheckExecutor()
        res = execu.execute_column_check(rule, s, attrs)
        assert 'leading_whitespace_ratio' in res.details
        assert 'trailing_whitespace_ratio' in res.details
        assert 'whitespace_only_ratio' in res.details
        assert 'non_printable_ratio' in res.details

