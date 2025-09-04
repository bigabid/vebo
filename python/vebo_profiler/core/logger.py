"""
Logging system for the data profiling process.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    """Log levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: LogLevel
    stage: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ProfilingLogger:
    """
    Logger for capturing detailed profiling progress.
    """
    
    def __init__(self):
        """Initialize the logger."""
        self.logs: List[LogEntry] = []
        self.current_stage = "initialization"
        self.stage_start_time = time.time()
    
    def set_stage(self, stage: str):
        """
        Set the current profiling stage.
        
        Args:
            stage: Current stage name
        """
        # Always log completion of previous stage (including initialization)
        if hasattr(self, 'current_stage') and self.current_stage:
            duration = time.time() - self.stage_start_time
            self.info(
                stage=self.current_stage, 
                message=f"Completed {self.current_stage} in {duration:.2f}s",
                details={"duration_seconds": duration}
            )
        
        self.current_stage = stage
        self.stage_start_time = time.time()
        self.info(stage=stage, message=f"Starting {stage}")
    
    def info(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Log an info message.
        
        Args:
            stage: Current stage
            message: Log message
            details: Optional additional details
        """
        self._add_log(LogLevel.INFO, stage, message, details)
    
    def warning(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Log a warning message.
        
        Args:
            stage: Current stage
            message: Log message
            details: Optional additional details
        """
        self._add_log(LogLevel.WARNING, stage, message, details)
    
    def error(self, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Log an error message.
        
        Args:
            stage: Current stage
            message: Log message
            details: Optional additional details
        """
        self._add_log(LogLevel.ERROR, stage, message, details)
    
    def _add_log(self, level: LogLevel, stage: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Add a log entry.
        
        Args:
            level: Log level
            stage: Current stage
            message: Log message
            details: Optional additional details
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            stage=stage,
            message=message,
            details=details
        )
        self.logs.append(entry)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get all logs as dictionaries.
        
        Returns:
            List of log dictionaries
        """
        return [
            {
                "timestamp": entry.timestamp,
                "level": entry.level.value,
                "stage": entry.stage,
                "message": entry.message,
                "details": entry.details
            }
            for entry in self.logs
        ]
    
    def get_latest_logs(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the latest N logs.
        
        Args:
            count: Number of logs to return
            
        Returns:
            List of latest log dictionaries
        """
        latest_logs = self.logs[-count:] if len(self.logs) > count else self.logs
        return [
            {
                "timestamp": entry.timestamp,
                "level": entry.level.value,
                "stage": entry.stage,
                "message": entry.message,
                "details": entry.details
            }
            for entry in latest_logs
        ]
    
    def clear_logs(self):
        """Clear all logs."""
        self.logs.clear()
