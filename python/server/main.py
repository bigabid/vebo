from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import os
import time
import threading
import uuid
import io
import math

import boto3
import pandas as pd

from vebo_profiler.core.profiler import VeboProfiler, ProfilingConfig
from vebo_profiler.core.logger import ProfilingLogger


class StartInsightsRequest(BaseModel):
    executionId: str
    table: str
    catalog: Optional[str] = None
    database: Optional[str] = None
    appliedFilters: Dict[str, list] = {}


class LogEntry(BaseModel):
    timestamp: str
    level: str  # info, warning, error
    stage: str  # sampling, column_analysis, cross_column_checks, etc.
    message: str
    details: Optional[Dict[str, Any]] = None

class InsightsJob(BaseModel):
    jobId: str
    status: str
    progress: Optional[int] = None  # Keep for backward compatibility
    message: Optional[str] = None
    insights: Optional[Dict[str, Any]] = None
    logs: List[LogEntry] = []


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
athena = boto3.client("athena", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)

# Simple in-memory job store
JOBS: Dict[str, InsightsJob] = {}


def _download_athena_csv_to_dataframe(s3_uri: str) -> pd.DataFrame:
    # s3_uri like s3://bucket/prefix/execid.csv
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI from Athena")
    path = s3_uri[5:]
    bucket, key = path.split("/", 1)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_csv(io.BytesIO(body))


def _map_profiler_to_insights(table: str, applied_filters: Dict[str, list], df: pd.DataFrame, profiler_output: Dict[str, Any]) -> Dict[str, Any]:
    # Basic mapping for the current UI expectations
    columns = []
    for col_name, analysis in profiler_output.get("column_analysis", {}).items():
        col_info = {
            "name": col_name,
            "type": analysis.get("data_type", "unknown"),
            "nullRatio": analysis.get("null_percentage", 0.0),
        }
        # numeric stats if available
        checks = analysis.get("checks", [])
        # index checks by id for convenience
        checks_by_id = {chk.get("check_id") or chk.get("rule_id"): chk for chk in checks if isinstance(chk, dict)}

        # Classification: categorical vs continuous
        data_type = str(analysis.get("data_type", "unknown"))
        unique_ratio = analysis.get("unique_percentage")
        unique_count = analysis.get("unique_count")
        value_type = None
        try:
            series = df[col_name]
        except Exception:
            series = None
        if data_type in ("categorical", "boolean", "textual", "array", "dictionary"):
            value_type = "categorical"
        elif data_type == "numeric":
            # Treat any float-valued numeric column as continuous
            is_float = False
            try:
                import pandas as _pd  # local alias
                if series is not None:
                    if _pd.api.types.is_float_dtype(series):
                        is_float = True
                    else:
                        num = _pd.to_numeric(series, errors='coerce')
                        non_null = num.dropna()
                        if len(non_null) > 0:
                            # Check for any fractional component
                            has_fractional = ((non_null % 1) != 0).any()
                            is_float = bool(has_fractional)
            except Exception:
                pass
            if is_float:
                value_type = "continuous"
            else:
                # Heuristic for integer-like numeric columns
                try:
                    if isinstance(unique_ratio, (int, float)) and unique_ratio is not None:
                        # Special case: 100% unique integer columns should be categorical (like ID columns)
                        if unique_ratio >= 0.99:  # Allow small floating point tolerance
                            value_type = "categorical"
                        else:
                            value_type = "continuous" if unique_ratio > 0.5 else "categorical"
                    elif isinstance(unique_count, (int, float)) and isinstance(analysis.get("total_count"), (int, float)):
                        ur = float(unique_count) / float(analysis.get("total_count") or 1)
                        # Special case: 100% unique integer columns should be categorical (like ID columns)
                        if ur >= 0.99:  # Allow small floating point tolerance
                            value_type = "categorical"
                        else:
                            value_type = "continuous" if ur > 0.5 else "categorical"
                    else:
                        value_type = "continuous"
                except Exception:
                    value_type = "continuous"
        elif data_type == "temporal":
            value_type = "continuous"
        if value_type:
            col_info["valueType"] = value_type

        # try to find numeric_stats details
        numeric_stats = None
        ns = checks_by_id.get("numeric_stats") or {}
        if isinstance(ns.get("details"), dict):
            numeric_stats = ns["details"].get("statistics")
        if numeric_stats:
            # Ensure min/max are converted to numbers (they come as strings from JSON)
            min_val = numeric_stats.get("min")
            max_val = numeric_stats.get("max")
            mean_val = numeric_stats.get("mean")
            
            col_info["numeric"] = {
                "min": float(min_val) if min_val is not None and str(min_val) != "" else None,
                "max": float(max_val) if max_val is not None and str(max_val) != "" else None,
                "avg": float(mean_val) if mean_val is not None and str(mean_val) != "" else None,
                # Add median as it's often more meaningful than mean for skewed distributions
                "median": float(numeric_stats.get("median")) if numeric_stats.get("median") is not None else None,
                # Add standard deviation for understanding data spread
                "std": float(numeric_stats.get("std")) if numeric_stats.get("std") is not None else None,
            }

        # Top values (from top_k_frequencies rule, which excludes nulls)
        top_k = checks_by_id.get("top_k_frequencies") or {}
        if isinstance(top_k.get("details"), dict):
            top_k_data = top_k["details"].get("top_k", [])
            if top_k_data and isinstance(top_k_data, list):
                # Convert to the format expected by UI
                col_info["topValues"] = [
                    {"value": str(item["value"]), "count": int(item["count"])}
                    for item in top_k_data[:10]  # Limit to top 10
                    if "value" in item and "count" in item
                ]
        
        # If no top_k_frequencies available, fall back to most_common_value (but exclude nulls)
        if "topValues" not in col_info:
            most_common = None
            mc = checks_by_id.get("most_common_value") or {}
            if isinstance(mc.get("details"), dict):
                most_common = mc.get("details")
            if most_common:
                val = most_common.get("most_common_value")
                cnt = most_common.get("frequency")
                # Only include if value is not null
                if val is not None and cnt is not None and not pd.isna(val):
                    col_info["topValues"] = [{"value": str(val), "count": int(cnt)}]

        # basic details: unique_count, duplicate_ratio, most common ratios, etc.
        basic: Dict[str, Any] = {}

        uc = checks_by_id.get("unique_count") or {}
        if isinstance(uc.get("details"), dict):
            if isinstance(uc["details"].get("unique_count"), (int, float)):
                basic["uniqueCount"] = uc["details"]["unique_count"]
            if isinstance(uc["details"].get("unique_ratio"), (int, float)):
                basic["uniqueRatio"] = uc["details"].get("unique_ratio")

        dup = checks_by_id.get("duplicate_value_analysis") or {}
        if isinstance(dup.get("details"), dict):
            basic["duplicateRatio"] = dup["details"].get("duplicate_ratio")
            if isinstance(dup["details"].get("duplicate_count"), (int, float)):
                basic["duplicateCount"] = dup["details"].get("duplicate_count")

        na = checks_by_id.get("null_analysis") or {}
        if isinstance(na.get("details"), dict):
            # Ensure null_count is converted to number (it comes as string from JSON)
            null_count = na["details"].get("null_count")
            basic["nullCount"] = int(null_count) if null_count is not None and str(null_count) != "" else None
            basic["nullRatioDetailed"] = na["details"].get("null_ratio")

        # Get most_common for basic details (separate from topValues)
        most_common = None
        mc = checks_by_id.get("most_common_value") or {}
        if isinstance(mc.get("details"), dict):
            most_common = mc.get("details")
        
        if isinstance(most_common, dict):
            # If most common value is null and we have a non-null alternative, prefer the non-null one
            most_common_value = most_common.get("most_common_value")
            if pd.isna(most_common_value) and most_common.get("most_common_non_null_value") is not None:
                # Use the most common non-null value for display
                basic["mostCommonValue"] = most_common.get("most_common_non_null_value")
                frequency = most_common.get("most_common_non_null_frequency")
                basic["mostCommonFrequency"] = int(frequency) if frequency is not None and str(frequency) != "" else None
                basic["mostCommonFrequencyRatio"] = most_common.get("most_common_non_null_frequency_ratio")
                basic["mostCommonValueNote"] = f"Most common non-null (most common overall is null)"
            else:
                # Use the regular most common value
                basic["mostCommonValue"] = most_common_value
                frequency = most_common.get("frequency")
                basic["mostCommonFrequency"] = int(frequency) if frequency is not None and str(frequency) != "" else None
                basic["mostCommonFrequencyRatio"] = most_common.get("frequency_ratio")
            
            # Add flag for constant columns
            if most_common.get("is_constant_column"):
                basic["isConstantColumn"] = True

        if basic:
            col_info["basic"] = basic

        # Text patterns (from text_patterns rule)
        text_patterns_check = checks_by_id.get("text_patterns") or {}
        if isinstance(text_patterns_check.get("details"), dict):
            patterns_data = text_patterns_check["details"]
            
            # Handle both old format (patterns) and new format (basic_patterns, inferred_patterns)
            basic_patterns = patterns_data.get("basic_patterns") or patterns_data.get("patterns", {})
            inferred_patterns = patterns_data.get("inferred_patterns", [])
            
            # Only add textPatterns if there's meaningful data
            if basic_patterns or inferred_patterns:
                text_patterns = {
                    "basic_patterns": basic_patterns,
                    "inferred_patterns": inferred_patterns,
                    "status": text_patterns_check.get("status", "unknown"),
                    "message": text_patterns_check.get("message", "")
                }
                col_info["textPatterns"] = text_patterns

        columns.append(col_info)

    insights = {
        "table": table,
        "appliedFilters": applied_filters,
        "rowCount": int(len(df)),
        "partitionSummary": {
            "partitionKeys": list(applied_filters.keys()),
            "selectedCount": sum(len(v) for v in applied_filters.values()),
            "totalDistinct": sum(len(v) for v in applied_filters.values()),
        },
        "columns": columns,
    }
    return insights


def _compute_candidate_keys(df: pd.DataFrame, table: str, max_pair_columns: int = 8) -> List[Dict[str, Any]]:
    """Suggest candidate unique keys based on uniqueness ratios.

    Returns list of { columns: [..], uniqueness: float, noNulls: bool }
    """
    candidates: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return candidates

    n = len(df)
    def _name_hint_score(cols: List[str]) -> Dict[str, Any]:
        score = 0.0
        hints: List[str] = []
        table_l = str(table or "").lower()
        for c in cols:
            cl = str(c).lower()
            if cl in ("id", "pk") or cl.endswith("_id") or cl.endswith("-id") or cl.endswith("id"):
                score += 0.2
                hints.append(f"name looks like id: {c}")
            if "uuid" in cl or "guid" in cl:
                score += 0.2
                hints.append(f"uuid/guid in name: {c}")
            if cl.endswith("key") or "_key" in cl or "key_" in cl or cl == "key":
                score += 0.1
                hints.append(f"key in name: {c}")
            if table_l and (cl == f"{table_l}_id" or (cl.startswith(table_l) and cl.endswith("id"))):
                score += 0.1
                hints.append(f"matches table-id pattern: {c}")
        # Prefer fewer columns
        if len(cols) > 1:
            score -= 0.05 * (len(cols) - 1)
        return {"score": max(0.0, min(1.0, score)), "hints": hints}

    # Single column uniqueness
    single_col_candidates = []
    for col in df.columns:
        # Count distinct excluding NaN
        series = df[col]
        distinct = series.nunique(dropna=True)
        null_ratio = float(series.isna().mean()) if len(df) else 0.0
        no_nulls = bool(null_ratio == 0.0)
        uniqueness = float(distinct) / float(n) if n else 0.0
        # Heuristics
        name_hint = _name_hint_score([str(col)])
        confidence = (uniqueness * 0.8 + (1.0 - null_ratio) * 0.2) + name_hint["score"]
        confidence = max(0.0, min(1.0, confidence))
        reason_bits = [f"uniqueness={uniqueness:.4f}", f"null_ratio={null_ratio:.4f}"] + name_hint["hints"]
        candidate = {
            "columns": [str(col)],
            "uniqueness": uniqueness,
            "noNulls": no_nulls,
            "confidence": confidence,
            "reason": "; ".join(reason_bits),
        }
        single_col_candidates.append(candidate)
        candidates.append(candidate)

    # Build lookup for single column confidences (threshold: 50%)
    # Use a threshold of 0.5 for high confidence
    single_col_confidence = {}
    for candidate in single_col_candidates:
        col_name = candidate["columns"][0]
        if candidate["confidence"] >= 0.5:
            single_col_confidence[col_name] = candidate["confidence"]

    # Pairwise uniqueness (limit number of columns to keep computation reasonable)
    # Skip combinations only if their confidence <= single column confidence
    cols_for_pairs = list(df.columns)[:max_pair_columns]
    for i in range(len(cols_for_pairs)):
        for j in range(i + 1, len(cols_for_pairs)):
            col1, col2 = cols_for_pairs[i], cols_for_pairs[j]
            
            subset = [col1, col2]
            distinct = len(df.drop_duplicates(subset=subset))
            null_ratio = float(df[subset].isna().any(axis=1).mean()) if len(df) else 0.0
            no_nulls = bool(null_ratio == 0.0)
            uniqueness = float(distinct) / float(n) if n else 0.0
            name_hint = _name_hint_score([str(subset[0]), str(subset[1])])
            confidence = (uniqueness * 0.8 + (1.0 - null_ratio) * 0.2) + name_hint["score"]
            confidence = max(0.0, min(1.0, confidence))
            
            # Skip combination only if it doesn't improve upon single column confidence
            col1_confidence = single_col_confidence.get(str(col1), 0.0)
            col2_confidence = single_col_confidence.get(str(col2), 0.0)
            max_single_confidence = max(col1_confidence, col2_confidence)
            
            if max_single_confidence > 0 and confidence <= max_single_confidence:
                continue
                
            reason_bits = [f"uniqueness={uniqueness:.4f}", f"null_ratio={null_ratio:.4f}"] + name_hint["hints"]
            candidates.append({
                "columns": [str(subset[0]), str(subset[1])],
                "uniqueness": uniqueness,
                "noNulls": no_nulls,
                "confidence": confidence,
                "reason": "; ".join(reason_bits),
            })

    # Sort: highest uniqueness first, prefer noNulls, prefer fewer columns
    candidates.sort(key=lambda c: (c.get("confidence", 0.0), c.get("uniqueness", 0.0), c.get("noNulls", False), -len(c.get("columns", []))), reverse=True)

    # Deduplicate by same column set
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for c in candidates:
        key = tuple(c["columns"])  # order-specific is fine here
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    # Keep top 10 to keep payload small
    return deduped[:10]


def _score_primary_key(cand: Dict[str, Any], df: pd.DataFrame, table: str) -> Dict[str, Any]:
    """Compute a confidence score for primary key using the provided checklist heuristics."""
    confidence = 0.0
    reasons: List[str] = []

    # 1) Uniqueness
    uniq = float(cand.get("uniqueness", 0.0))
    if uniq >= 0.9999:
        confidence += 0.35
        reasons.append(f"uniqueness ~100%: {uniq:.4f}")
    elif uniq >= 0.999:
        confidence += 0.25
        reasons.append(f"uniqueness high: {uniq:.4f}")

    # 2) Completeness (Non-NULL)
    if cand.get("noNulls", False):
        confidence += 0.15
        reasons.append("no nulls")

    # 3) Stability Over Time (approx via low change in candidate cardinality across simple splits if partition/date columns exist)
    # If there is a date-like column, quickly check cardinality consistency across first two values
    date_cols = [c for c in df.columns if str(c).lower() in ("date", "dt", "event_date")]
    if date_cols:
        col = date_cols[0]
        try:
            sample_vals = list(pd.Series(df[col]).dropna().astype(str).head(200).unique())[:2]
            if len(sample_vals) == 2:
                d1 = df[df[col].astype(str) == sample_vals[0]]
                d2 = df[df[col].astype(str) == sample_vals[1]]
                key = cand["columns"]
                u1 = len(d1.drop_duplicates(subset=key)) if len(d1) else 0
                u2 = len(d2.drop_duplicates(subset=key)) if len(d2) else 0
                if u1 and u2 and abs(u1 - u2) / max(u1, u2) < 0.05:
                    confidence += 0.1
                    reasons.append("stable across dates (approx)")
        except Exception:
            pass

    # 4) Minimality: penalize if any proper subset also nearly-unique
    if len(cand.get("columns", [])) > 1:
        subset_almost_unique = False
        for i in range(len(cand["columns"])):
            sub = cand["columns"][:i] + cand["columns"][i+1:]
            if not sub:
                continue
            distinct = len(df.drop_duplicates(subset=sub))
            if len(df) and (float(distinct) / float(len(df))) >= 0.999:
                subset_almost_unique = True
                break
        if not subset_almost_unique:
            confidence += 0.1
            reasons.append("minimal (no subset unique)")

    # 5) Business invariance (name heuristics)
    table_l = str(table).lower()
    name_str = "|".join([str(c).lower() for c in cand.get("columns", [])])
    if any(k in name_str for k in ("id", "uuid", "guid", "pk", "key")):
        confidence += 0.1
        reasons.append("name suggests identifier")
    if table_l and any(name_str.startswith(f"{table_l}_") or name_str.endswith(f"_{table_l}") for _ in [0]):
        confidence += 0.05
        reasons.append("name aligns with table")

    # 8) Distribution & Skew (penalize if too few distinct)
    distinct_over_rows = uniq
    if distinct_over_rows < 0.5:
        confidence -= 0.1
        reasons.append("low cardinality relative to rows")

    # 9) Format validity (simple regex/length hints)
    try:
        import re
        if len(cand.get("columns", [])) == 1:
            c = cand["columns"][0]
            series = pd.Series(df[c]).dropna().astype(str).head(200)
            if series.str.match(r"^[0-9a-fA-F-]{36}$").mean() > 0.8:
                confidence += 0.05
                reasons.append("uuid-like")
            if series.str.match(r"^\d{6,}$").mean() > 0.8:
                confidence += 0.03
                reasons.append("numeric-id-like")
    except Exception:
        pass

    # Clip and attach
    cand_pk = dict(cand)
    cand_pk["confidence"] = max(0.0, min(1.0, confidence))
    cand_pk["reason"] = "; ".join(reasons)
    return cand_pk


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    try:
        import numpy as np  # type: ignore
        if isinstance(value, (np.generic,)):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _process_job(job_id: str, execution_id: str, table: str, applied_filters: Dict[str, list]):
    # Create logger for this job
    logger = ProfilingLogger()
    
    try:
        logger.set_stage("athena_polling")
        logger.info(
            stage="athena_polling", 
            message=f"Starting Athena query polling for execution {execution_id}",
            details={"execution_id": execution_id, "table": table}
        )
        
        # Update job with initial logs
        JOBS[job_id] = InsightsJob(
            jobId=job_id, 
            status="running", 
            progress=0,
            logs=[LogEntry(**log) for log in logger.get_logs()]
        )
        
        # Poll Athena for completion
        for attempt in range(120):  # ~10 minutes max
            resp = athena.get_query_execution(QueryExecutionId=execution_id)
            status = resp.get("QueryExecution", {}).get("Status", {}).get("State")
            
            if status == "SUCCEEDED":
                logger.info(
                    stage="athena_polling",
                    message="Athena query completed successfully, downloading results"
                )
                
                # Update job with download progress
                JOBS[job_id] = InsightsJob(
                    jobId=job_id, 
                    status="running", 
                    progress=20,
                    logs=[LogEntry(**log) for log in logger.get_logs()]
                )
                
                logger.set_stage("data_download")
                output_loc = resp["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
                logger.info(
                    stage="data_download",
                    message=f"Downloading query results from {output_loc}"
                )
                
                df = _download_athena_csv_to_dataframe(output_loc)
                logger.info(
                    stage="data_download",
                    message=f"Downloaded dataset with {len(df)} rows and {len(df.columns)} columns",
                    details={"rows": len(df), "columns": len(df.columns)}
                )
                
                # Update job with profiling start
                JOBS[job_id] = InsightsJob(
                    jobId=job_id, 
                    status="running", 
                    progress=30,
                    logs=[LogEntry(**log) for log in logger.get_logs()]
                )
                
                # Start profiling with the shared logger
                logger.set_stage("profiling_start")
                logger.info(
                    stage="profiling_start",
                    message="Starting data profiling process"
                )
                
                profiler = VeboProfiler(ProfilingConfig(), logger=logger)
                
                # Add some debug logging
                logger.info(
                    stage="profiling_start",
                    message="Background log updater starting..."
                )
                
                # Debug: Check initial log count
                initial_logs = logger.get_logs()
                print(f"Initial log count before starting profiler: {len(initial_logs)}")
                
                # Set up periodic log updates during profiling
                def periodic_log_updater():
                    iteration = 0
                    while job_id in JOBS:
                        try:
                            iteration += 1
                            current_job = JOBS.get(job_id)
                            if current_job and current_job.status == "running":
                                # Get fresh logs from the shared logger
                                raw_logs = logger.get_logs()
                                fresh_logs = [LogEntry(**log) for log in raw_logs]
                                log_count = len(fresh_logs)
                                
                                # Debug: Print to console
                                print(f"Background updater iteration {iteration}: Found {log_count} logs")
                                if log_count > 0:
                                    latest_raw_log = raw_logs[-1]
                                    print(f"Latest log: {latest_raw_log.get('message', 'No message')}")
                                
                                # Update job with fresh logs
                                JOBS[job_id] = InsightsJob(
                                    jobId=job_id,
                                    status="running",
                                    progress=current_job.progress or 50,
                                    message=current_job.message,
                                    logs=fresh_logs
                                )
                            elif current_job and current_job.status in ["cancelled", "complete", "error"]:
                                print(f"Background updater stopping: job status is {current_job.status}")
                                break
                            else:
                                print("Background updater stopping: job not found or invalid")
                                break
                        except Exception as e:
                            print(f"Background log updater error: {e}")
                            import traceback
                            traceback.print_exc()
                        time.sleep(2)  # Update every 2 seconds
                    print("Background log updater thread exiting")
                
                # Start the background log updater
                log_updater_thread = threading.Thread(target=periodic_log_updater, daemon=True)
                log_updater_thread.start()
                
                # Check for cancellation before starting profiling
                current_job = JOBS.get(job_id)
                if current_job and current_job.status == "cancelled":
                    return
                
                profiling_result = profiler.profile_dataframe(df, filename=table)
                
                # Update job progress during post-processing
                logger.set_stage("post_processing")
                logger.info(
                    stage="post_processing",
                    message="Processing profiling results and generating insights"
                )
                
                JOBS[job_id] = InsightsJob(
                    jobId=job_id, 
                    status="running", 
                    progress=80,
                    logs=[LogEntry(**log) for log in logger.get_logs()]
                )
                
                # convert dataclass to dict via to_json or asdict
                profiling_json = profiler.to_json(profiling_result)
                data = json.loads(profiling_json)
                insights = _map_profiler_to_insights(table, applied_filters, df, data)

                # Attach cross-column checks (raw)
                try:
                    cc_checks = data.get("cross_column_analysis", {}).get("checks", [])
                except Exception:
                    cc_checks = []
                # Normalize list of checks to a simpler structure
                cross_results = []
                for chk in cc_checks:
                    if not isinstance(chk, dict):
                        continue
                    details = chk.get("details") if isinstance(chk.get("details"), dict) else {}
                    compared = details.get("compared_columns") if isinstance(details.get("compared_columns"), dict) else {}
                    cross_results.append({
                        "checkId": chk.get("check_id") or chk.get("rule_id"),
                        "name": chk.get("name"),
                        "status": chk.get("status"),
                        "message": chk.get("message"),
                        "columns": [compared.get("column_1"), compared.get("column_2")] if compared else None,
                        "details": details,
                    })
                insights["crossColumn"] = cross_results
                
                logger.info(
                    stage="post_processing",
                    message="Computing candidate keys and primary key suggestions"
                )
                
                # Attach candidate unique keys suggestions
                candidates = _compute_candidate_keys(df, table)
                insights["candidateKeys"] = candidates
                # Compute primary key scores from candidates
                insights["primaryKeys"] = [_score_primary_key(c, df, table) for c in candidates[:10]]
                insights = _sanitize_for_json(insights)
                
                logger.info(
                    stage="post_processing",
                    message="All processing completed successfully",
                    details={
                        "candidate_keys": len(candidates),
                        "primary_keys": min(10, len(candidates))
                    }
                )
                
                JOBS[job_id] = InsightsJob(
                    jobId=job_id, 
                    status="complete", 
                    insights=insights,
                    logs=[LogEntry(**log) for log in logger.get_logs()]
                )
                return
                
            elif status in ("FAILED", "CANCELLED"):
                msg = resp.get("QueryExecution", {}).get("Status", {}).get("StateChangeReason")
                logger.error(
                    stage="athena_polling",
                    message=f"Athena query failed: {msg}",
                    details={"status": status, "reason": msg}
                )
                JOBS[job_id] = InsightsJob(
                    jobId=job_id, 
                    status="error", 
                    message=msg,
                    logs=[LogEntry(**log) for log in logger.get_logs()]
                )
                return
            else:
                # QUEUED or RUNNING
                progress = min(15, attempt // 2)  # Cap initial progress at 15%
                if attempt % 6 == 0:  # Update logs every 30 seconds (6 * 5s)
                    logger.info(
                        stage="athena_polling",
                        message=f"Athena query still running... (attempt {attempt + 1}/120)",
                        details={"status": status, "attempt": attempt + 1}
                    )
                
                JOBS[job_id] = InsightsJob(
                    jobId=job_id, 
                    status="running", 
                    progress=progress,
                    logs=[LogEntry(**log) for log in logger.get_logs()]
                )
                time.sleep(5)
                
        # Timeout
        logger.error(
            stage="athena_polling",
            message="Timed out waiting for Athena result after 10 minutes"
        )
        JOBS[job_id] = InsightsJob(
            jobId=job_id, 
            status="error", 
            message="Timed out waiting for Athena result",
            logs=[LogEntry(**log) for log in logger.get_logs()]
        )
        
    except Exception as e:
        logger.error(
            stage="error",
            message=f"Unexpected error during job processing: {str(e)}",
            details={"error_type": type(e).__name__, "error_message": str(e)}
        )
        JOBS[job_id] = InsightsJob(
            jobId=job_id, 
            status="error", 
            message=str(e),
            logs=[LogEntry(**log) for log in logger.get_logs()]
        )


@app.post("/insights/start")
def start_insights(req: StartInsightsRequest, background_tasks: BackgroundTasks):
    if not req.executionId:
        raise HTTPException(status_code=400, detail="Missing executionId")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = InsightsJob(jobId=job_id, status="running", progress=0)
    background_tasks.add_task(_process_job, job_id, req.executionId, req.table, req.appliedFilters or {})
    return {"jobId": job_id, "status": "running"}


@app.get("/insights/status")
def get_insights_status(jobId: str):
    job = JOBS.get(jobId)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=_sanitize_for_json(job.dict()))


class CancelJobRequest(BaseModel):
    jobId: str

@app.post("/insights/cancel")
def cancel_insights(req: CancelJobRequest):
    """Cancel a running insights job."""
    job_id = req.jobId
    if not job_id:
        raise HTTPException(status_code=400, detail="Missing jobId")
    
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in ["running"]:
        raise HTTPException(status_code=400, detail="Job is not running")
    
    # Mark job as cancelled
    JOBS[job_id] = InsightsJob(
        jobId=job_id,
        status="cancelled",
        message="Job cancelled by user",
        logs=job.logs if hasattr(job, 'logs') else []
    )
    
    return {"status": "cancelled", "message": "Job cancelled successfully"}


@app.get("/health")
def health():
    return {"ok": True, "region": AWS_REGION}


