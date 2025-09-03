"""
Check executor for running data profiling checks.
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .rule_engine import Rule, RuleResult, RuleStatus
from .meta_rules import ColumnAttributes, MetaRuleDetector


@dataclass
class ExecutionConfig:
    """Configuration for check execution."""
    enable_cross_column: bool = True
    max_workers: int = None
    timeout_seconds: int = 300
    enable_parallel: bool = True
    sample_size: int = 10000
    sampling_threshold: int = 100000


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
            # Generate and execute the check code
            check_code = self._generate_check_code(rule, series, attributes)
            result_dict = self._execute_check_code(check_code, series)
            
            # Convert result to RuleResult
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus(result_dict.get("status", "error")),
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
                message=f"Check execution failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                execution_time_ms=execution_time,
                timestamp=timestamp
            )
    
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
            # Generate and execute the check code
            check_code = self._generate_cross_column_check_code(rule, col1, col2)
            result_dict = self._execute_cross_column_check_code(check_code, df, col1, col2)
            
            # Convert result to RuleResult
            execution_time = (time.time() - start_time) * 1000
            
            return RuleResult(
                rule_id=rule.id,
                rule_name=rule.name,
                status=RuleStatus(result_dict.get("status", "error")),
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
                message=f"Cross-column check execution failed: {str(e)}",
                details={"error": str(e), "traceback": traceback.format_exc()},
                execution_time_ms=execution_time,
                timestamp=timestamp
            )
    
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
        namespace = {
            'pd': pd,
            'np': np,
            'series': series
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
        Execute multiple cross-column checks in parallel.
        
        Args:
            rules: List of rules to execute
            df: DataFrame to check
            column_pairs: List of column pairs to check
            
        Returns:
            List of RuleResult objects
        """
        if not self.config.enable_parallel:
            # Execute sequentially
            results = []
            for rule in rules:
                for col1, col2 in column_pairs:
                    result = self.execute_cross_column_check(rule, df, col1, col2)
                    results.append(result)
            return results
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for rule in rules:
                for col1, col2 in column_pairs:
                    future = executor.submit(self.execute_cross_column_check, rule, df, col1, col2)
                    futures.append(future)
            
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
                        message=f"Parallel cross-column execution failed: {str(e)}",
                        details={"error": str(e)},
                        execution_time_ms=0.0,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(error_result)
            
            return results
