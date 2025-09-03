from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
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


