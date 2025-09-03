from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import json
import os
import time
import uuid
import io
import math

import boto3
import pandas as pd

from vebo_profiler.core.profiler import VeboProfiler, ProfilingConfig


class StartInsightsRequest(BaseModel):
    executionId: str
    table: str
    catalog: Optional[str] = None
    database: Optional[str] = None
    appliedFilters: Dict[str, list] = {}


class InsightsJob(BaseModel):
    jobId: str
    status: str
    progress: Optional[int] = None
    message: Optional[str] = None
    insights: Optional[Dict[str, Any]] = None


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
        # try to find numeric_stats details
        numeric_stats = None
        for chk in checks:
            if chk.get("check_id") == "numeric_stats" and isinstance(chk.get("details"), dict):
                numeric_stats = chk["details"].get("statistics")
                break
        if numeric_stats:
            col_info["numeric"] = {
                "min": numeric_stats.get("min"),
                "max": numeric_stats.get("max"),
                "avg": numeric_stats.get("mean"),
            }
        # common values
        most_common = None
        for chk in checks:
            if chk.get("check_id") == "most_common_value":
                most_common = chk.get("details")
                break
        if most_common:
            val = most_common.get("most_common_value")
            cnt = most_common.get("frequency")
            if val is not None and cnt is not None:
                col_info["topValues"] = [{"value": str(val), "count": int(cnt)}]
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
        candidates.append({
            "columns": [str(col)],
            "uniqueness": uniqueness,
            "noNulls": no_nulls,
            "confidence": confidence,
            "reason": "; ".join(reason_bits),
        })

    # Pairwise uniqueness (limit number of columns to keep computation reasonable)
    cols_for_pairs = list(df.columns)[:max_pair_columns]
    for i in range(len(cols_for_pairs)):
        for j in range(i + 1, len(cols_for_pairs)):
            subset = [cols_for_pairs[i], cols_for_pairs[j]]
            distinct = len(df.drop_duplicates(subset=subset))
            null_ratio = float(df[subset].isna().any(axis=1).mean()) if len(df) else 0.0
            no_nulls = bool(null_ratio == 0.0)
            uniqueness = float(distinct) / float(n) if n else 0.0
            name_hint = _name_hint_score([str(subset[0]), str(subset[1])])
            confidence = (uniqueness * 0.8 + (1.0 - null_ratio) * 0.2) + name_hint["score"]
            confidence = max(0.0, min(1.0, confidence))
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
    # Poll Athena for completion
    try:
        for attempt in range(120):  # ~10 minutes max
            resp = athena.get_query_execution(QueryExecutionId=execution_id)
            status = resp.get("QueryExecution", {}).get("Status", {}).get("State")
            if status == "SUCCEEDED":
                output_loc = resp["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
                df = _download_athena_csv_to_dataframe(output_loc)
                profiler = VeboProfiler(ProfilingConfig())
                profiling_result = profiler.profile_dataframe(df, filename=table)
                # convert dataclass to dict via to_json or asdict
                profiling_json = profiler.to_json(profiling_result)
                data = json.loads(profiling_json)
                insights = _map_profiler_to_insights(table, applied_filters, df, data)
                # Attach candidate unique keys suggestions
                candidates = _compute_candidate_keys(df, table)
                insights["candidateKeys"] = candidates
                # Compute primary key scores from candidates
                insights["primaryKeys"] = [_score_primary_key(c, df, table) for c in candidates[:10]]
                insights = _sanitize_for_json(insights)
                JOBS[job_id] = InsightsJob(jobId=job_id, status="complete", insights=insights)
                return
            elif status in ("FAILED", "CANCELLED"):
                msg = resp.get("QueryExecution", {}).get("Status", {}).get("StateChangeReason")
                JOBS[job_id] = InsightsJob(jobId=job_id, status="error", message=msg)
                return
            else:
                # QUEUED or RUNNING
                JOBS[job_id] = InsightsJob(jobId=job_id, status="running", progress=min(99, attempt))
                time.sleep(5)
        # Timeout
        JOBS[job_id] = InsightsJob(jobId=job_id, status="error", message="Timed out waiting for Athena result")
    except Exception as e:
        JOBS[job_id] = InsightsJob(jobId=job_id, status="error", message=str(e))


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


@app.get("/health")
def health():
    return {"ok": True, "region": AWS_REGION}


