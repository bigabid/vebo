"""
Main profiler class that orchestrates the data profiling process.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import time

from .meta_rules import MetaRuleDetector, ColumnAttributes
from .rule_engine import RuleEngine, RuleResult
from .check_executor import CheckExecutor, ExecutionConfig
from .logger import ProfilingLogger


@dataclass
class ProfilingConfig:
    """Configuration for data profiling."""
    enable_cross_column: bool = True
    deepness_level: str = "standard"  # basic, standard, deep
    max_workers: int = None
    timeout_seconds: int = 300
    sample_size: int = 10000
    sampling_threshold: int = 100000
    random_seed: int = 42


@dataclass
class ProfilingResult:
    """Result of data profiling."""
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    column_analysis: Dict[str, Any]
    cross_column_analysis: Dict[str, Any]
    table_level_analysis: Dict[str, Any]
    errors: List[Dict[str, Any]]


class VeboProfiler:
    """
    Main profiler class that orchestrates the data profiling process.
    """
    
    def __init__(self, config: ProfilingConfig = None, logger: ProfilingLogger = None):
        """
        Initialize the profiler.
        
        Args:
            config: Profiling configuration
            logger: Optional logger for capturing profiling progress
        """
        self.config = config or ProfilingConfig()
        self.logger = logger or ProfilingLogger()
        self.meta_detector = MetaRuleDetector(seed=self.config.random_seed)
        self.rule_engine = RuleEngine()
        
        # Create execution config
        execution_config = ExecutionConfig(
            enable_cross_column=self.config.enable_cross_column,
            max_workers=self.config.max_workers,
            timeout_seconds=self.config.timeout_seconds,
            enable_parallel=True,
            sample_size=self.config.sample_size,
            sampling_threshold=self.config.sampling_threshold
        )
        
        self.check_executor = CheckExecutor(execution_config)
        
        self.logger.info(
            stage="initialization",
            message="VeboProfiler initialized",
            details={
                "deepness_level": self.config.deepness_level,
                "cross_column_enabled": self.config.enable_cross_column,
                "max_workers": self.config.max_workers
            }
        )
    
    def profile_dataframe(self, df: pd.DataFrame, filename: str = None) -> ProfilingResult:
        """
        Profile a pandas DataFrame.
        
        Args:
            df: DataFrame to profile
            filename: Optional filename for metadata
            
        Returns:
            ProfilingResult object
        """
        start_time = time.time()
        
        self.logger.set_stage("data_preparation")
        self.logger.info(
            stage="data_preparation",
            message=f"Starting data profiling for dataset with {len(df)} rows and {len(df.columns)} columns",
            details={
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        )
        
        # Determine if sampling is needed
        self.logger.set_stage("sampling_decision")
        if self.meta_detector.should_enable_sampling(df, self.config.sampling_threshold):
            self.logger.info(
                stage="sampling_decision",
                message=f"Dataset size ({len(df)} rows) exceeds threshold ({self.config.sampling_threshold}), creating sample",
                details={"original_size": len(df), "sample_size": self.config.sample_size}
            )
            df_to_analyze = self.meta_detector.create_sample(df, self.config.sample_size)
            was_sampled = True
            self.logger.info(
                stage="sampling_decision",
                message=f"Sample created with {len(df_to_analyze)} rows",
                details={"sampled_rows": len(df_to_analyze)}
            )
        else:
            self.logger.info(
                stage="sampling_decision",
                message="No sampling needed, analyzing full dataset",
                details={"rows": len(df)}
            )
            df_to_analyze = df
            was_sampled = False
        
        # Analyze column attributes
        self.logger.set_stage("column_analysis")
        self.logger.info(
            stage="column_analysis", 
            message="Analyzing column attributes and data types"
        )
        column_attributes = self.meta_detector.analyze_dataframe(df_to_analyze)
        self.logger.info(
            stage="column_analysis",
            message=f"Completed column analysis for {len(column_attributes)} columns",
            details={
                "analyzed_columns": len(column_attributes),
                "data_types": {col: attr.type_category.value for col, attr in list(column_attributes.items())[:5]}  # Show first 5
            }
        )
        
        # Execute column-level checks
        self.logger.set_stage("column_checks")
        self.logger.info(
            stage="column_checks",
            message="Executing column-level data quality checks"
        )
        column_results = self._execute_column_checks(df_to_analyze, column_attributes)
        total_column_checks = sum(len(results) for results in column_results.values())
        self.logger.info(
            stage="column_checks",
            message=f"Completed {total_column_checks} column-level checks across {len(column_results)} columns",
            details={"total_checks": total_column_checks, "columns_analyzed": len(column_results)}
        )
        
        # Execute cross-column checks if enabled
        cross_column_results = []
        if self.config.enable_cross_column and self.config.deepness_level in ["standard", "deep"]:
            self.logger.set_stage("cross_column_checks")
            self.logger.info(
                stage="cross_column_checks",
                message="Executing cross-column relationship checks"
            )
            cross_column_results = self._execute_cross_column_checks(df_to_analyze, column_attributes)
            self.logger.info(
                stage="cross_column_checks",
                message=f"Completed {len(cross_column_results)} cross-column checks",
                details={"cross_column_checks": len(cross_column_results)}
            )
        else:
            self.logger.info(
                stage="column_checks",
                message="Cross-column checks skipped (disabled or basic deepness level)"
            )
        
        # Execute table-level checks
        self.logger.set_stage("table_checks")
        self.logger.info(
            stage="table_checks",
            message="Executing table-level data quality checks"
        )
        table_results = self._execute_table_checks(df_to_analyze, column_attributes)
        self.logger.info(
            stage="table_checks",
            message=f"Completed {len(table_results)} table-level checks",
            details={"table_checks": len(table_results)}
        )
        
        # Compile results
        self.logger.set_stage("results_compilation")
        self.logger.info(
            stage="results_compilation",
            message="Compiling profiling results and generating report"
        )
        end_time = time.time()
        duration = end_time - start_time
        
        result = self._compile_results(
            df, df_to_analyze, filename, was_sampled, 
            column_attributes, column_results, cross_column_results, 
            table_results, start_time, end_time, duration
        )
        
        self.logger.info(
            stage="results_compilation",
            message=f"Data profiling completed successfully in {duration:.2f} seconds",
            details={
                "total_duration": duration,
                "was_sampled": was_sampled,
                "total_checks": total_column_checks + len(cross_column_results) + len(table_results)
            }
        )
        
        return result
    
    def _execute_column_checks(self, df: pd.DataFrame, 
                             column_attributes: Dict[str, ColumnAttributes]) -> Dict[str, List[RuleResult]]:
        """
        Execute column-level checks for all columns.
        
        Args:
            df: DataFrame to analyze
            column_attributes: Column attributes
            
        Returns:
            Dictionary mapping column names to their check results
        """
        column_results = {}
        
        for col_name, attributes in column_attributes.items():
            # Get relevant rules for this column
            relevant_rules = self.rule_engine.get_relevant_rules(
                attributes, 
                enable_cross_column=False  # Column-level checks only
            )
            
            if not relevant_rules:
                continue
            
            # Execute checks for this column
            series = df[col_name]
            results = self.check_executor.execute_column_checks_parallel(
                relevant_rules, series, attributes
            )
            
            column_results[col_name] = results
        
        return column_results
    
    def _execute_cross_column_checks(self, df: pd.DataFrame, 
                                   column_attributes: Dict[str, ColumnAttributes]) -> List[RuleResult]:
        """
        Execute cross-column checks.
        
        Args:
            df: DataFrame to analyze
            column_attributes: Column attributes
            
        Returns:
            List of cross-column check results
        """
        # Get cross-column rules
        cross_column_rules = self.rule_engine.get_rules_by_category("cross_column")
        
        if not cross_column_rules:
            return []
        
        # Generate column pairs for analysis
        column_pairs = self._generate_column_pairs(df, column_attributes)
        
        if not column_pairs:
            return []
        
        # Execute cross-column checks
        results = self.check_executor.execute_cross_column_checks_parallel(
            cross_column_rules, df, column_pairs
        )
        
        return results
    
    def _execute_table_checks(self, df: pd.DataFrame, 
                            column_attributes: Dict[str, ColumnAttributes]) -> List[RuleResult]:
        """
        Execute table-level checks.
        
        Args:
            df: DataFrame to analyze
            column_attributes: Column attributes
            
        Returns:
            List of table-level check results
        """
        # For now, we'll implement basic table-level checks
        # This can be expanded with more sophisticated table-level rules
        
        table_results = []
        
        # Basic table statistics
        table_stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Create a simple table-level result
        from .rule_engine import RuleResult, RuleStatus
        
        table_result = RuleResult(
            rule_id="table_stats",
            rule_name="Table Statistics",
            status=RuleStatus.PASSED,
            score=100.0,
            message=f"Table has {table_stats['total_rows']} rows and {table_stats['total_columns']} columns",
            details=table_stats,
            execution_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        table_results.append(table_result)
        
        return table_results
    
    def _generate_column_pairs(self, df: pd.DataFrame, 
                             column_attributes: Dict[str, ColumnAttributes]) -> List[tuple]:
        """
        Generate column pairs for cross-column analysis.
        
        Args:
            df: DataFrame to analyze
            column_attributes: Column attributes
            
        Returns:
            List of column pairs
        """
        column_pairs = []
        columns = list(df.columns)
        
        # Generate all possible pairs
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                
                # Check if both columns have sufficient data
                attr1 = column_attributes.get(col1)
                attr2 = column_attributes.get(col2)
                
                if attr1 and attr2:
                    # Only include pairs where both columns have some data
                    if attr1.null_count < attr1.total_count and attr2.null_count < attr2.total_count:
                        column_pairs.append((col1, col2))
        
        return column_pairs
    
    def _compile_results(self, original_df: pd.DataFrame, analyzed_df: pd.DataFrame, 
                        filename: str, was_sampled: bool, column_attributes: Dict[str, ColumnAttributes],
                        column_results: Dict[str, List[RuleResult]], cross_column_results: List[RuleResult],
                        table_results: List[RuleResult], start_time: float, end_time: float, 
                        duration: float) -> ProfilingResult:
        """
        Compile all results into a ProfilingResult object.
        
        Args:
            original_df: Original DataFrame
            analyzed_df: DataFrame that was actually analyzed (may be sampled)
            filename: Optional filename
            was_sampled: Whether sampling was used
            column_attributes: Column attributes
            column_results: Column-level check results
            cross_column_results: Cross-column check results
            table_results: Table-level check results
            start_time: Analysis start time
            end_time: Analysis end time
            duration: Analysis duration
            
        Returns:
            ProfilingResult object
        """
        # Compile metadata
        metadata = {
            "dataset_info": {
                "filename": filename or "unknown",
                "rows": len(original_df),
                "columns": len(original_df.columns),
                "file_size_bytes": original_df.memory_usage(deep=True).sum(),
                "sampling_info": {
                    "was_sampled": was_sampled,
                    "sample_size": len(analyzed_df) if was_sampled else len(original_df),
                    "sample_method": "random" if was_sampled else "none",
                    "seed": self.config.random_seed
                }
            },
            "execution_info": {
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": duration,
                "rules_processed": len(self.rule_engine.list_rules()),
                "checks_executed": sum(len(results) for results in column_results.values()) + 
                                 len(cross_column_results) + len(table_results),
                "errors_encountered": sum(1 for results in column_results.values() 
                                        for result in results if result.status.value == "error") +
                                    sum(1 for result in cross_column_results if result.status.value == "error") +
                                    sum(1 for result in table_results if result.status.value == "error")
            },
            "configuration": {
                "enabled_categories": ["basic_stats", "numeric_stats", "text_patterns", "cross_column"],
                "deepness_level": self.config.deepness_level,
                "cross_column_checks": self.config.enable_cross_column
            }
        }
        
        # Compile summary
        all_results = []
        for results in column_results.values():
            all_results.extend(results)
        all_results.extend(cross_column_results)
        all_results.extend(table_results)
        
        passed_count = sum(1 for result in all_results if result.status.value == "passed")
        failed_count = sum(1 for result in all_results if result.status.value == "failed")
        warning_count = sum(1 for result in all_results if result.status.value == "warning")
        error_count = sum(1 for result in all_results if result.status.value == "error")
        
        overall_score = (passed_count * 100 + warning_count * 75 + failed_count * 25) / len(all_results) if all_results else 0
        
        if overall_score >= 90:
            quality_grade = "A"
        elif overall_score >= 80:
            quality_grade = "B"
        elif overall_score >= 70:
            quality_grade = "C"
        elif overall_score >= 60:
            quality_grade = "D"
        else:
            quality_grade = "F"
        
        summary = {
            "overall_score": overall_score,
            "quality_grade": quality_grade,
            "critical_issues": failed_count,
            "warnings": warning_count,
            "recommendations": len(all_results) - error_count
        }
        
        # Compile column analysis
        column_analysis = {}
        for col_name, attributes in column_attributes.items():
            results = column_results.get(col_name, [])
            
            column_analysis[col_name] = {
                "data_type": attributes.type_category.value,
                "null_count": attributes.null_count,
                "null_percentage": attributes.null_count / attributes.total_count if attributes.total_count > 0 else 0,
                "unique_count": attributes.unique_count,
                "unique_percentage": attributes.unique_count / attributes.total_count if attributes.total_count > 0 else 0,
                "checks": [self._result_to_dict(result) for result in results],
                "visualizations": []  # TODO: Add visualization generation
            }
        
        # Compile cross-column analysis
        cross_column_analysis = {
            "correlations": [],  # TODO: Extract correlation results
            "checks": [self._result_to_dict(result) for result in cross_column_results],
            "visualizations": []  # TODO: Add visualization generation
        }
        
        # Compile table-level analysis
        table_level_analysis = {
            "checks": [self._result_to_dict(result) for result in table_results],
            "visualizations": []  # TODO: Add visualization generation
        }
        
        # Compile errors
        errors = []
        for result in all_results:
            if result.status.value == "error":
                errors.append({
                    "check_id": result.rule_id,
                    "error_type": "execution_error",
                    "message": result.message,
                    "timestamp": result.timestamp,
                    "severity": "high"
                })
        
        return ProfilingResult(
            metadata=metadata,
            summary=summary,
            column_analysis=column_analysis,
            cross_column_analysis=cross_column_analysis,
            table_level_analysis=table_level_analysis,
            errors=errors
        )
    
    def _result_to_dict(self, result: RuleResult) -> Dict[str, Any]:
        """Convert RuleResult to dictionary."""
        return {
            "check_id": result.rule_id,
            "rule_id": result.rule_id,
            "name": result.rule_name,
            "description": "",  # TODO: Add description to Rule class
            "status": result.status.value,
            "score": result.score,
            "message": result.message,
            "details": result.details,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp
        }
    
    def to_json(self, result: ProfilingResult) -> str:
        """
        Convert ProfilingResult to JSON string.
        
        Args:
            result: ProfilingResult object
            
        Returns:
            JSON string
        """
        return json.dumps(asdict(result), indent=2, default=str)
    
    def save_result(self, result: ProfilingResult, filename: str):
        """
        Save ProfilingResult to a JSON file.
        
        Args:
            result: ProfilingResult object
            filename: Output filename
        """
        json_str = self.to_json(result)
        with open(filename, 'w') as f:
            f.write(json_str)
