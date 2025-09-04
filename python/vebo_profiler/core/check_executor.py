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
            # Generate the check code
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
        Execute multiple cross-column checks in parallel with skip logic for identical columns.
        
        Args:
            rules: List of rules to execute
            df: DataFrame to check
            column_pairs: List of column pairs to check
            
        Returns:
            List of RuleResult objects
        """
        # First, find and execute identicality rule to identify identical column pairs
        identicality_rule = None
        other_rules = []
        
        for rule in rules:
            if rule.id == "identicality":
                identicality_rule = rule
            else:
                other_rules.append(rule)
        
        results = []
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
        
        # Filter column pairs to exclude identical ones for other rules
        non_identical_pairs = [pair for pair in column_pairs if pair not in identical_pairs]
        
        if not other_rules or not non_identical_pairs:
            return results
        
        if not self.config.enable_parallel:
            # Execute sequentially for non-identical pairs
            for rule in other_rules:
                for col1, col2 in non_identical_pairs:
                    result = self.execute_cross_column_check(rule, df, col1, col2)
                    results.append(result)
            return results
        
        # Execute other rules in parallel for non-identical pairs only
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for rule in other_rules:
                for col1, col2 in non_identical_pairs:
                    future = executor.submit(self.execute_cross_column_check, rule, df, col1, col2)
                    futures.append(future)
            
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
