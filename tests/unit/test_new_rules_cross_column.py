"""
Unit tests for new cross-column rules.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from vebo_profiler.core.rule_engine import RuleEngine
from vebo_profiler.core.check_executor import CheckExecutor


class TestNewCrossColumnRules:
    def setup_method(self):
        self.engine = RuleEngine()
        self.execu = CheckExecutor()

    def test_missingness_relationships(self):
        df = pd.DataFrame({
            'a': [1, None, 3, None],
            'b': [None, 2, None, 4]
        })
        rule = self.engine.get_rule('missingness_relationships')
        res = self.execu.execute_cross_column_check(rule, df, 'a', 'b')
        assert 'both_null_count' in res.details

    def test_functional_dependency(self):
        df = pd.DataFrame({
            'k': ['x','x','y','y','z'],
            'v': [1,1,2,2,3]
        })
        rule = self.engine.get_rule('functional_dependency')
        res = self.execu.execute_cross_column_check(rule, df, 'k', 'v')
        assert 'fd_holds_ratio' in res.details

    def test_composite_uniqueness(self):
        df = pd.DataFrame({
            'c1': [1,1,2,2],
            'c2': ['a','a','b','c']
        })
        rule = self.engine.get_rule('composite_uniqueness')
        res = self.execu.execute_cross_column_check(rule, df, 'c1', 'c2')
        assert 'unique_pair_ratio' in res.details

    def test_inclusion_dependency(self):
        # Ensure equal length columns
        df = pd.DataFrame({
            'left': ['a','b','c','c'],
            'right': ['a','b','c','d']
        })
        rule = self.engine.get_rule('inclusion_dependency')
        res = self.execu.execute_cross_column_check(rule, df, 'left', 'right')
        assert 'coverage_ratio' in res.details

    def test_categorical_association_cramers_v(self):
        df = pd.DataFrame({
            'cat1': ['A','A','B','B','C','C'],
            'cat2': ['X','X','Y','Y','Z','Z']
        })
        rule = self.engine.get_rule('categorical_association_cramers_v')
        res = self.execu.execute_cross_column_check(rule, df, 'cat1', 'cat2')
        assert 'cramers_v' in res.details

